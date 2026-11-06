import json
import base64
import io
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from headway.animation import STAGES, Walkthrough, export_html, export_downloads


@pytest.fixture(scope="module")
def animation_results(tmp_path_factory):
    root = tmp_path_factory.mktemp("animation")
    subprocess.run([sys.executable, "run.py", "--dataset", "highd", "--iters", "600",
                    "--warmup", "300", "--synthetic-size", "400", "--workers", "1",
                    "--data-dir", str(root / "no_raw_data"), "--out", str(root)],
                   check=True, capture_output=True)
    return root


def test_replay_and_every_animation_stage(animation_results):
    walk = Walkthrough(animation_results, "highd")
    assert len(walk.replay["headway_preview"]) == 200
    assert walk.replay["simulated_headways"].shape == (400,)
    assert np.all(np.diff(walk.replay["renewal_counts"], axis=1) >= 0)
    assert np.allclose(walk.replay["markov_transient"][1], walk.P[0])
    # Every stage must render at its initial, intermediate and final state.
    for stage in range(len(STAGES)):
        for progress in (0, 0.5, 1):
            walk.frame(stage, progress).canvas.draw()
            assert walk.description
    plt.close(walk.fig)


def test_offline_export_and_older_results(animation_results, tmp_path):
    walk = Walkthrough(animation_results, "highd", model="Burr")
    # Old outputs lack replay arrays: explanations must say so rather than fake paths.
    walk.replay = {}
    walk.frame(10, 1)
    assert any("No arrival paths saved" in text.get_text() for text in walk.fig.axes[0].texts)
    destination = export_html(walk, tmp_path / "demo.html", frames_per_stage=2)
    html = destination.read_text(encoding="utf-8")
    payload = json.loads(html.split("const data = ", 1)[1].split(";\nconst image", 1)[0])
    assert len(payload["frames"]) == 2 * len(STAGES)
    assert len(payload["descriptions"]) == len(payload["frames"])
    assert payload["source"] == "synthetic"
    assert "__ANIMATION_DATA__" not in html
    assert "https://" not in html and 'src="http' not in html
    assert all(frame.startswith("iVBOR") for frame in payload["frames"])


def test_mismatched_histogram_is_rejected(animation_results, tmp_path):
    folder = tmp_path / "highd"
    folder.mkdir()
    info = pd.read_csv(animation_results / "highd" / "dataset_info.csv")
    info.to_csv(folder / "dataset_info.csv", index=False)
    pd.DataFrame({"count": np.ones(49)}).to_csv(folder / "headway_histogram.csv", index=False)
    with pytest.raises(ValueError, match="matching dataset_info"):
        Walkthrough(tmp_path, "highd")


@pytest.mark.skipif(shutil.which("node") is None, reason="Optional JavaScript control check needs Node")
def test_offline_player_controls():
    # Execute the real player script with a minimal DOM; no browser/package dependency.
    html = Path("headway/animation_player.html").read_text(encoding="utf-8")
    script = html.split("<script>")[1].split("</script>")[0]
    data = {"frames": ["cG5n"] * 6, "descriptions": ["explanation"] * 6, "gif": "R0lGODlh",
            "stages": ["Data", "Fit", "Simulation"], "framesPerStage": 2,
            "dataset": "highd", "source": "synthetic"}
    setup = """
const assert = require('node:assert/strict');
const elements = new Map();
const downloads = [];
let keyHandler, tick;
global.document = {
  documentElement: {outerHTML:'<html>offline player</html>'},
  body: {appendChild() {}},
  createElement() { return {click() {downloads.push(this.download);}, remove(){}}; },
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, {value:'1', options:[],
      add(option) { this.options.push(option); }, click() { this.onclick(); }});
    return elements.get(id);
  },
  addEventListener(name, handler) { keyHandler = handler; }
};
global.Option = function(text, value) { this.text = text; this.value = value; };
global.setInterval = callback => { tick = callback; return 1; };
global.clearInterval = () => {};
global.setTimeout = callback => callback();
"""
    checks = """
assert.equal(current, 0);
assert.equal(stage.options.length, 3);
assert.equal(document.getElementById('explanation').textContent, 'explanation');
document.getElementById('next').click(); assert.equal(current, 2);
document.getElementById('previous').click(); assert.equal(current, 0);
stage.value = 2; stage.onchange(); assert.equal(current, 4);
play.click(); assert.equal(play.textContent, 'Pause');
tick(); assert.equal(current, 5); assert.equal(timer, null);
play.click(); assert.equal(current, 0);
seek.value = 3; seek.oninput(); assert.equal(current, 3); assert.equal(timer, null);
keyHandler({target:{tagName:'BODY'}, code:'Home', preventDefault(){}});
assert.equal(current, 0);
keyHandler({target:{tagName:'BODY'}, code:'Space', preventDefault(){}});
assert.equal(play.textContent, 'Pause');
speed.value = 2; speed.onchange(); tick(); assert.equal(current, 1);
assert.equal(document.getElementById('download-gif').hidden, false);
document.getElementById('download-html').click();
document.getElementById('download-png').click();
document.getElementById('download-gif').click();
assert.deepEqual(downloads, ['highd-animation.html', 'highd-frame-2.png', 'highd-animation.gif']);
"""
    subprocess.run(["node", "-e", setup + script.replace("__ANIMATION_DATA__", json.dumps(data)) + checks],
                   check=True, capture_output=True)


def test_gif_export_preserves_frames_timing_and_embeds_download(tmp_path):
    frames = []
    for color in ("red", "blue"):
        buffer = io.BytesIO()
        Image.new("RGB", (16, 16), color).save(buffer, format="PNG")
        frames.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
    payload = {"frames": frames, "descriptions": ["a", "b"], "stages": ["Example"],
               "framesPerStage": 2, "dataset": "highd", "source": "synthetic"}
    path = tmp_path / "example.html"
    template = Path("headway/animation_player.html").read_text(encoding="utf-8")
    path.write_text(template.replace("__ANIMATION_DATA__", json.dumps(payload)), encoding="utf-8")
    gif_path = export_downloads(path)
    with Image.open(gif_path) as gif:
        assert gif.n_frames == 2 and gif.info["loop"] == 0
        assert gif.info["duration"] == 2000
        assert gif.convert("RGB").getpixel((0, 0)) == (255, 0, 0)
        gif.seek(1)
        assert gif.convert("RGB").getpixel((0, 0)) == (0, 0, 255)
    updated = path.read_text(encoding="utf-8")
    saved = json.loads(updated.split("const data = ", 1)[1].split(";\nconst image", 1)[0])
    assert base64.b64decode(saved["gif"]) == gif_path.read_bytes()
    assert saved["frames"] == frames
