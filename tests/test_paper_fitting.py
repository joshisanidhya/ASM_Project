"""Independent checks of Table 1 likelihoods and boundary-sensitive posterior sampling."""
import numpy as np
import pytest
from scipy import stats

from headway.diagnostics import effective_sample_size, split_rhat
from headway.distributions import ALPHA, DISTS, EDGES, T_MAX
from headway.fit import ParameterTransform, fit_all, make_logpost
from headway.paper_reference import PAPER_TABLE3_PARAMS, PAPER_TABLE5_KL, PAPER_TABLE5_W, compare_with_paper
from headway.metrics import histogram_distances


@pytest.mark.parametrize("dataset", ["highd", "ngsim"])
@pytest.mark.parametrize("name", list(DISTS))
def test_observation_likelihood_matches_independent_scipy_pdfs(dataset, name):
    theta = PAPER_TABLE3_PARAMS[dataset][name]
    data = np.array([0.65, 0.8, 1.3, 1.3, 2.7, 4.0, 7.5, 20.0])
    if name == "Proposed":
        a, b = theta
        ref = stats.laplace(a, 1 / -np.log(b))
        logs = ref.logpdf(data) - ref.logsf(ALPHA)
    elif name == "Shifted Log-normal":
        mu, sigma, shift = theta
        logs = stats.lognorm.logpdf(data, sigma, loc=shift, scale=np.exp(mu))
    elif name == "Weibull":
        logs = stats.weibull_min.logpdf(data, theta[0], scale=theta[1])
    elif name == "Log-logistic":
        logs = stats.fisk.logpdf(data, theta[0], scale=theta[1])
    elif name == "Gamma":
        logs = stats.gamma.logpdf(data, theta[0], scale=1 / theta[1])
    elif name == "Burr":
        logs = stats.burr12.logpdf(data, theta[0], theta[1], scale=theta[2])
    else:
        logs = stats.expon.logpdf(data, loc=theta[1], scale=1 / theta[0])
    expected = logs.sum() + DISTS[name].logprior(theta, data.min())
    assert make_logpost(DISTS[name], data)(theta) == pytest.approx(expected, abs=1e-10)
    values, frequencies = np.unique(data, return_counts=True)
    assert make_logpost(DISTS[name], values, frequencies)(theta) == pytest.approx(expected, abs=1e-10)


def test_actual_minimum_sets_shift_support_not_half_second_cutoff():
    for name in ("Shifted Exponential", "Shifted Log-normal"):
        theta = PAPER_TABLE3_PARAMS["ngsim"][name]
        assert DISTS[name].check_params(theta)
        assert np.isfinite(make_logpost(DISTS[name], [0.6, 1.4, 2.5])(theta))
        assert make_logpost(DISTS[name], [0.5, 1.4, 2.5])(theta) == -np.inf
        assert np.isclose(DISTS[name].bin_probs(theta).sum(), 1)


def test_proposed_below_cutoff_reduces_to_shifted_exponential():
    data = np.array([0.5, 0.6, 0.9, 5.0])
    for a in (-1000.0, 0.2, 0.5):
        theta = (a, 0.6)
        rate = -np.log(theta[1])
        expected = stats.expon.logpdf(data, loc=ALPHA, scale=1 / rate).sum()
        assert make_logpost(DISTS["Proposed"], data)(theta) == pytest.approx(
            expected + DISTS["Proposed"].logprior(theta, data.min()))
        assert np.allclose(DISTS["Proposed"].pdf(data, *theta), stats.expon.pdf(data, loc=ALPHA, scale=1 / rate))


def test_transform_jacobian_matches_numerical_derivative():
    transform = ParameterTransform([(-np.inf, np.inf), (0, np.inf), (0, 1), (-10, 0.6)])
    theta = np.array([2.0, 1.3, 0.54, 0.558])
    z = transform.forward(theta)
    recovered, logdet = transform.inverse(z)
    assert np.allclose(recovered, theta)
    eps = np.eye(4) * 1e-5
    jac = np.column_stack([(transform.inverse(z + e)[0] - transform.inverse(z - e)[0]) / 2e-5 for e in eps])
    assert logdet == pytest.approx(np.linalg.slogdet(jac)[1], abs=1e-7)


def test_shifted_exponential_million_observation_posterior():
    # Match the reported highD sample scale; the posterior width of the shift is O(1/n).
    data = 0.6 + np.random.default_rng(19).exponential(1 / 0.584, 1_225_460)
    data = data[data <= T_MAX]
    fit = fit_all(data, names=["Shifted Exponential"], workers=1)["Shifted Exponential"]
    n, minimum = len(data), data.min()
    # Integrating gamma over (-10, min(data)) gives this Gamma marginal, up to
    # exp(-n*lambda*(min(data)+10)), negligible here. Conditional min(data)-gamma
    # is Exp(n*lambda) with the same negligible truncation.
    shape = n - 0.5
    rate = np.sum(data - minimum) + 0.5
    expected_lambda = shape / rate
    posterior_sd = np.sqrt(shape) / rate
    expected_gap = rate / (n * (shape - 1))
    assert abs(fit.theta[0] - expected_lambda) < 0.2 * posterior_sd
    assert (minimum - fit.theta[1]) == pytest.approx(expected_gap, rel=0.12)
    assert np.all(fit.draws[:, :, 1] < minimum)
    assert np.all(split_rhat(fit.draws) < 1.05)
    assert np.all(effective_sample_size(fit.draws) > 200)


@pytest.mark.parametrize("data", [[], [1.0], [0.2, 2.0], [1.0, np.nan], [1.0, 26.0]])
def test_fit_rejects_invalid_observations(data):
    with pytest.raises(ValueError, match="headways"):
        fit_all(data, workers=1)


def test_stuck_chains_cannot_pass_diagnostics():
    for draws in (np.ones((2, 500, 1)), np.broadcast_to([[[1.0]], [[2.0]]], (2, 500, 1))):
        assert np.isinf(split_rhat(draws)).all()
        assert np.all(effective_sample_size(draws) == 0)
    # Very small but nonzero posterior widths must remain measurable.
    draws = np.random.default_rng(1).normal(size=(2, 2000, 1)) * 1e-9
    assert split_rhat(draws)[0] < 1.05
    assert effective_sample_size(draws)[0] > 1000


def test_paper_comparison_keeps_reference_values_out_of_fits():
    params = PAPER_TABLE3_PARAMS["highd"]
    counts = 1000 * DISTS["Proposed"].bin_probs(params["Proposed"])
    metrics = [{"distribution": name, **histogram_distances(counts, DISTS[name].bin_probs(theta))}
               for name, theta in params.items()]
    diagnostics = [{"distribution": name, "converged": True} for name in params]
    rows = compare_with_paper("highd", counts, params, metrics, diagnostics)
    for row in rows:
        assert row["fitted_kl"] == pytest.approx(row["table3_on_current_data_kl"])
        assert row["table5_kl"] == PAPER_TABLE5_KL["highd"][row["distribution"]]
    # This published pair cannot result from the legacy sorted-bin metric on 49
    # normalized probabilities with natural-log KL. It must not be a tuning target.
    bound = np.sqrt(2 * PAPER_TABLE5_KL["ngsim"]["Gamma"]) / (len(EDGES) - 1)
    assert PAPER_TABLE5_W["ngsim"]["Gamma"] > bound
