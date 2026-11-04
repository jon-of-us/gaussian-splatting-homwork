
= Embodied AI \ Homework 1

\

Franz Jonas Ullmann \ 林阳 \ 2026400694

= Task 0: Environment setup and baseline rendering

== Environment and result

The assignment was completed using Google Colab with a GPU runtime.

The main errors encountered were:

- `std::uintptr_t` was undeclared in `rasterizer_impl.h` → CUDA rasterizer compilation failed.
- `uint32_t` and `uint64_t` were undeclared in `rasterizer_impl.h` → CUDA rasterizer compilation failed.

== Baseline artifact

The notebook packages the five views and environment details as `task0_artifacts.zip`. Extract the archive into this `report/` directory to make the images and `task0_environment.txt` available locally.

#figure(
  image("images/task0_baseline.png", width: 70%),
  caption: [Baseline render from camera 0, produced by the unfinished starter implementation.],
)

= Task 1: Dataset preparation

== COLMAP sparse reconstruction

I ran the fruit images through COLMAP. Feature extraction processed all 220 images (220/220), with no images skipped at that stage. The reconstruction pipeline shown in the log had these stages:

1. *Feature extraction:* COLMAP detected SIFT features in each image (about 3,200--3,900 per image).
2. *Feature matching and geometric verification:* COLMAP matched features between images and removes matches that are inconsistent with the estimated geometry. #text(fill: blue)[0.27 minutes]
3. *Incremental mapping:* COLMAP initialized a reconstruction and registered images successively, triangulating 3D points from the verified matches. #text(fill: blue)[11.90 minutes]
4. *Retriangulation and global bundle adjustment:* COLMAP refined the points, camera poses, and camera parameters as more images were registered. #text(fill: blue)[3.94 minutes]

 It showed linear-solver warnings during optimization, but finished by keeping a successful reconstruction. 
//  The log reports about 0.27 minutes for feature extraction, 3.94 minutes for feature matching and geometric verification, and 11.90 minutes for mapping.

For camera initialization, I selected _Parameters from EXIF_ in the GUI. The log nevertheless assigns a different `SIMPLE_RADIAL` camera ID to each image Their reported focal length is 1726.80 pixels, consistent with COLMAP's default initialization for that image size rather than a shared camera calibration. I kept this reconstruction because the camera groups are unknown; consequently, its camera models were not shared across images.

== 3D point-cloud visualization

The COLMAP viewer shows the registered camera poses around the reconstructed scene. I also opened the provided `points3D.ply` point cloud in MeshLab.

#figure(
  image("images/colmap-screenshot.png", width: 75%),
  caption: [Sparse reconstruction and registered camera poses in the COLMAP 3D viewer.],
)

#figure(
  image("images/meshlab-screenshot.png", width: 75%),
  caption: [The provided `points3D.ply` point cloud viewed in MeshLab (35,674 vertices).],
)

= Task 2: Building Gaussian primitives

== Covariance from scale and rotation

Let the activated scales be standard deviations along a Gaussian's local principal axes, and let $R$ be the rotation represented by its normalized quaternion. The diagonal scale matrix is

$
D = mat(s_x, 0, 0; 0, s_y, 0; 0, 0, s_z).
$

The linear map from a unit isotropic Gaussian to this oriented Gaussian is $L = R D$. Covariance transforms by multiplying the map by its transpose:

$
Sigma = L L^T = R D D^T R^T
       = R mat(s_x^2, 0, 0; 0, s_y^2, 0; 0, 0, s_z^2) R^T.
$

In `get_covariance`, the stored log-scales are exponentiated by `get_scaling`, the quaternion is normalized and converted to $R$, and `scaling_modifier` multiplies each scale before forming $L$. The returned six values are the unique entries of the symmetric covariance matrix.

== Viewing transformation

Write the affine world-to-camera transform as

$
T_v = mat(R_v, t_v; 0, 1),
$

where $R_v$ is its $3 times 3$ linear part and $t_v$ is its translation. For a world-space Gaussian with mean $mu$ and covariance $Sigma$, the camera-space mean and covariance are

$
mu_c = R_v mu + t_v, quad Sigma_c = R_v Sigma R_v^T.
$

Translation changes the mean but not the covariance. In homogeneous coordinates a point has final component $1$; a direction has final component $0$.

== Local projection and screen-space covariance

For camera-space coordinates $(x, y, z)$ and focal lengths $f_x, f_y$ in pixels, perspective projection (omitting the principal-point offset, which has zero derivative) is

$
u = f_x x / z, quad v = f_y y / z.
$

Linearizing this nonlinear map at the camera-space mean gives the Jacobian

$
J_p = mat(
  f_x/z, 0, -f_x x/z^2;
  0, f_y/z, -f_y y/z^2
).
$

The first-order projected Gaussian has mean equal to the projection of $mu_c$ and covariance

$
Sigma_(2D) = J_p Sigma_c J_p^T.
$

Equivalently, with $A = J_p R_v$, $Sigma_(2D) = A Sigma A^T$. This order first changes the covariance from world to camera coordinates and then applies the projection Jacobian. `computeCov2D` implements this propagation using the repository's column-major GLM convention: it embeds the transposed Jacobian in a $3 times 3$ matrix, combines it with the reconstructed view rotation, and returns the symmetric upper-left $2 times 2$ block. The code adds $0.3$ to both diagonal entries as an isotropic screen-space low-pass filter, so the returned diagonal values are in pixel-squared units.

== Square enclosing the 99% joint confidence region

For a two-dimensional Gaussian, the squared Mahalanobis distance follows a chi-squared distribution with two degrees of freedom. Thus the 99% confidence ellipse is

$
(p - mu_(2D))^T Sigma_(2D)^(-1) (p - mu_(2D)) <= q,
quad q = chi^2_(2, 0.99) = -2 ln(0.01) approx 9.21034.
$

Writing the diagonal entries of $Sigma_(2D)$ as $sigma_(u u)$ and $sigma_(v v)$, the ellipse's greatest absolute extent along screen axis $u$ is $sqrt(q sigma_(u u))$, and along $v$ it is $sqrt(q sigma_(v v))$. Therefore, the smallest axis-aligned square centered at the mean that contains the entire ellipse has side length

$
ell = 2 sqrt(q max(sigma_(u u), sigma_(v v))).
$

This square contains the 99% joint ellipse, so its probability coverage can exceed 99%. The largest covariance eigenvalue would instead bound the ellipse's radius in every direction and can make this axis-aligned square unnecessarily large; a 1D normal quantile or independent marginal-coverage argument does not in general describe the joint 99% region.

== Rendered result

#figure(
  image("images/task2-render.png", width: 65%),
  caption: [Representative view rendered after implementing the Task 2 covariance calculations.],
)

= Task 3: Spherical-harmonic degree and opacity pruning

== Experimental setup

I ran the comparison notebook on a Google Colab Tesla T4 GPU with Python 3.13.16, PyTorch 2.11.0+cu130, and CUDA 13.0. Each SH setting rendered all five evaluation cameras. Image error was measured against the supplied ground-truth views using RGB mean absolute error (MAE) and peak signal-to-noise ratio (PSNR). Three alternating-order runs were used for each SH degree; the reported render time is the median wall-clock time for the complete five-view `render.py` process.

== SH degree comparison

Degree 1 improved both image metrics on every camera. Averaged over the five views, MAE decreased from 0.02327 to 0.01849 (20.6% lower), while PSNR increased from 29.17 dB to 31.56 dB. The quality improvement came with a modest timing increase: median time rose from 5.92 s for degree 0 to 6.31 s for degree 1 (6.6%).

#table(
  columns: 4,
  table.header([SH degree], [Mean MAE], [Mean PSNR], [Median five-view time]),
  [0], [0.02327], [29.17 dB], [5.92 s],
  [1], [0.01849], [31.56 dB], [6.31 s],
)

#figure(
  image("images/task3_sh_comparison/view_comparison.png", width: 100%),
  caption: [Ground truth and rendered views for SH degrees 0 and 1, across the five evaluation cameras.],
)

#figure(
  image("images/task3_sh_comparison/image_metrics.png", width: 95%),
  caption: [Per-camera MAE and PSNR; degree 1 improves both metrics for all five views.],
)

#figure(
  image("images/task3_sh_comparison/render_times.png", width: 80%),
  caption: [Five-view render wall time over three trials; bars show medians and points show individual runs.],
)

== Opacity-based post-hoc pruning

I pruned the degree-1 model without retraining, using six minimum-opacity thresholds. The initial model contained 257,882 Gaussians. The most aggressive tested threshold, 0.10, removed 88,225 Gaussians (34.2%); the five-view contact sheet remained broadly similar to the unpruned output, although small details may be affected.

#table(
  columns: 4,
  table.header([Opacity threshold], [Gaussians pruned], [Pruned fraction], [Five-view time]),
  [0.000], [0], [0.0%], [6.47 s],
  [0.005], [23,865], [9.3%], [5.75 s],
  [0.010], [30,199], [11.7%], [6.29 s],
  [0.020], [41,232], [16.0%], [5.67 s],
  [0.050], [63,961], [24.8%], [6.51 s],
  [0.100], [88,225], [34.2%], [5.64 s],
)

#figure(
  image("images/task3_sh_comparison/pruning_fractions.png", width: 90%),
  caption: [Fraction of the original Gaussians removed at each opacity threshold.],
)

#figure(
  image("images/task3_sh_comparison/threshold_0.100.png", width: 100%),
  caption: [Unpruned baseline (top) and degree-1 render after pruning at opacity threshold 0.10 (bottom), across all five cameras.],
)

The pruning timings are single runs and are not monotonic with threshold. They include process startup, model loading, pruning, rendering, and writing all five images, so the observed times do not establish a reliable speedup from pruning. The threshold sweep instead shows the trade-off between the number of removed Gaussians and the visual quality of the rendered views.
