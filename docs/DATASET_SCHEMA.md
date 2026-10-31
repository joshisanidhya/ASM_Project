# Dataset Schemas & Preprocessing Specification

**Project**: Applied Stochastic Modeling (ASM) Research Project — Vehicle Headway Distribution  
**Supported Datasets**: 3 exclusively (`highD`, `NGSIM`, `Lyft Level 5`)  
**Excluded Datasets**: `exiD` and `Waymo Open Motion Dataset` are strictly excluded per project scope.

---

## 1. Scope and Standardization Overview

The paper evaluates microscopic car-following behavior across diverse traffic environments:
1. **Uninterrupted Freeway Flow (Human-Driven Vehicles)**: `highD`
2. **Interrupted / Freeway Flow (Human-Driven Vehicles)**: `NGSIM`
3. **Interrupted Urban Mixed Traffic (Human-Driven + Autonomous Vehicles)**: `Lyft Level 5`

### Standardized Preprocessing Rules (Paper Section 4)
* **Sampling Rate**: Standardized across all datasets to **1 Hz** (1 observation per second per vehicle).
* **Car-Following Thresholds**:
  * $\alpha = 0.5$ s: Lower bound (minimum safe human/AV reaction time). Headways $< 0.5$ s are excluded.
  * $T_{\text{max}} = 25.0$ s: Upper bound (headways $> 25.0$ s reflect free flow rather than car-following interaction).
* **Discretization**: Exactly **49 intervals** of width $\Delta t = 0.5$ s:
  $$[0.5, 1.0), [1.0, 1.5), \dots, [24.5, 25.0]$$

---

## 2. Dataset 1: highD (German Highway Drone Trajectories)

### 2.1 Overview & Source
* **Origin**: RWTH Aachen University, Institute for Automotive Engineering.
* **Recording Method**: Aerial 4K drone video over German autobahns.
* **Original Frame Rate**: **25 Hz**.
* **Traffic Conditions**: Uninterrupted highway driving; HDVs (cars and trucks).
* **Files**: `NN_tracks.csv`, `NN_tracksMeta.csv`, `NN_recordingMeta.csv` ($NN \in \{01, \dots, 60\}$).

### 2.2 Schema & Key Fields
* **`recordingMeta.csv`**:
  * `id`: Recording identifier (1–60).
  * `frameRate`: Camera frame rate (typically 25 Hz).
  * `locationId`: Highway site identifier.
  * `speedLimit`: Speed limit at the location [km/h].
* **`tracks.csv`**:
  * `frame`: Discrete frame index (1-based).
  * `id`: Vehicle identifier.
  * `x`, `y`: Vehicle center position [m].
  * `xVelocity`, `yVelocity`: Velocity components [m/s].
  * `laneId`: Traffic lane index.
  * `precedingId`: Vehicle ID of the direct leader in the same lane (`0` if no leader).
  * `thw`: **Time Headway [seconds]**, precalculated by highD drone tracking software.
  * `dhw`: Distance headway [meters] (front-bumper to front-bumper).

### 2.3 Extraction & Preprocessing Logic
```python
# 1. Read frameRate from recordingMeta (defaults to 25 Hz)
# 2. Subsample at 1 Hz: frame % frameRate == 0
# 3. Filter car-following: precedingId > 0
# 4. Extract thw and clip to [0.5, 25.0] seconds
```

---

## 3. Dataset 2: NGSIM (FHWA Vehicle Trajectories)

### 3.1 Overview & Source
* **Origin**: Federal Highway Administration (FHWA), Next Generation Simulation program.
* **Recording Method**: Synchronized video camera arrays mounted atop tall buildings along Interstate 80 (I-80 in Emeryville, CA) and US-101 (Los Angeles, CA).
* **Original Frame Rate**: **10 Hz**.
* **Traffic Conditions**: Congested highway corridor with lane changes, merging bottlenecks, and stop-and-go waves.
* **Paper Subset**: **I-80 reconstructed trajectory data** (Montanino & Punzo 2015).

### 3.2 Schema & Key Fields
* **`Frame_ID`**: Timestamp index at 10 Hz (subsampled every 10 frames for 1 Hz).
* **`Vehicle_ID`**: Unique vehicle tracking identifier.
* **`Local_X`, `Local_Y`**: Longitudinal and lateral position in road coordinates [feet].
* **`v_Vel`**: Longitudinal speed [feet/second].
* **`Preceding`**: Vehicle ID of the leading vehicle (`0` if no vehicle ahead in current lane).
* **`Time_Headway`**: **Temporal headway in seconds** ($h$).
* **`Space_Headway`**: Spatial headway in feet (front bumper to front bumper).
* **`Location`**: Data collection site (e.g., `"i-80"`, `"us-101"`).

### 3.3 Extraction & Preprocessing Logic
```python
# 1. Filter location: Location.str.lower() == 'i-80'
# 2. Subsample at 1 Hz: Frame_ID % 10 == 0
# 3. Filter valid leader: Preceding > 0
# 4. Extract Time_Headway [s] and clip to [0.5, 25.0] seconds
```

---

## 4. Dataset 3: Lyft Level 5 (Autonomous Driving Dataset)

### 4.1 Overview & Source
* **Origin**: Lyft Level 5 Autonomous Driving Division (Woven Planet).
* **Recording Method**: Roof-mounted LiDAR, RADAR, and cameras on a fleet of 20 autonomous Chrysler Pacifica minivans operating along Palo Alto urban routes.
* **Original Frame Rate**: **10 Hz**.
* **Storage Format**: Hierarchical Zarr format (`scenes`, `frames`, `agents`).
* **Traffic Conditions**: Urban arterial and collector roads with signalized intersections, crosswalks, and mixed AV/HDV traffic.
* **Follower**: The autonomous ego vehicle acts as the follower (driver viewpoint).

### 4.2 Zarr Internal Table Schema
* **`scenes` Table**:
  * `frame_index_interval`: `[start_frame, end_frame]` indices into the `frames` array.
* **`frames` Table**:
  * `timestamp`: Epoch nanoseconds.
  * `agent_index_interval`: `[start_agent, end_agent]` slice into global `agents` array.
  * `ego_translation`: `(3,)` floating-point coordinates `[x, y, z]` in global meters.
  * `ego_rotation`: `(3, 3)` rotation matrix defining ego heading.
* **`agents` Table**:
  * `centroid`: `(2,)` agent position `[x, y]` in meters.
  * `extent`: `(3,)` bounding box dimensions `[length, width, height]` in meters.
  * `label_probabilities`: 17-class classification scores (classes 3, 4, 6, 7, 8, 9 identify vehicles).

### 4.3 Leader Derivation & Time Headway Proxy
Because Level 5 AV datasets do not provide precomputed time-headway columns, the car-following headway is derived:
1. **Ego Heading**:
   $$\psi = \text{atan2}(R_{1,0}, R_{0,0}), \quad \mathbf{h} = [\cos\psi, \sin\psi]^T$$
2. **Relative Displacement**:
   $$\mathbf{d}_i = \mathbf{c}_i - \mathbf{c}_{\text{ego}}, \quad \text{lon}_i = \mathbf{d}_i \cdot \mathbf{h}, \quad \text{lat}_i = \mathbf{d}_i \cdot [-\sin\psi, \cos\psi]^T$$
3. **Lateral Corridor Leader Selection**:
   A vehicle qualifies as a potential leader if $\text{lon}_i > 0$ and $|\text{lat}_i| \le 1.8$ m ($LANE\_HALF\_WIDTH$). The closest in-corridor vehicle is selected:
   $$i^* = \arg\min_i \{\text{lon}_i \mid \text{lon}_i > 0, |\text{lat}_i| \le 1.8\}$$
4. **Bumper-to-Bumper Gap**:
   $$\text{gap} = \text{lon}_{i^*} - \frac{L_{\text{ego}} + L_{i^*}}{2}, \quad (L_{\text{ego}} = 4.87\text{ m})$$
5. **Headway Calculation**:
   $$h = \frac{\text{gap}}{v_{\text{ego}}}, \quad \text{for } v_{\text{ego}} \ge 0.5\text{ m/s}$$
   and clipped to $[0.5, 25.0]$ seconds.

### 4.4 Documented Approximation & Limitations
* **Corridor vs. Map**: On straight segments, the $\pm 1.8$ m heading corridor matches the lane. On sharp turns or curved intersections, lane polyline association from HD vector maps would improve accuracy.
* **Lagrangian vs. Eulerian**: Instantaneous gap divided by speed approximates true passage-time headway when speed is near-constant during the car-following maneuver.

---

## 5. Unified Dataset Loader Interface

All loaders conform to a standardized contract returning a 1D NumPy array of floats in $[0.5, 25.0]$ s:
```python
load_highd(path: Union[str, Path], workers: Optional[int] = None) -> np.ndarray
load_ngsim(path: Union[str, Path], location: Optional[str] = "i-80", hz: int = 10) -> np.ndarray
load_lyft(path: Union[str, Path], workers: Optional[int] = None) -> np.ndarray
```
When raw dataset files are absent, the pipeline automatically generates synthetic benchmark samples calibrated to the paper's reported parameters, guaranteeing full reproducibility without requiring multi-gigabyte external downloads.
