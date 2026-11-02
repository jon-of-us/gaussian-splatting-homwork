# Embodied AI Homework 1 — 3DGS

**Release date:** Oct 15, 2026  
**Deadline date:** Nov 5, 2026, 23:59 (Beijing time, UTC+08:00)  
**Goal:** Familiarize yourself with 3D Gaussian Splatting (3DGS) principles and workflows.

You have **three weeks** to complete this assignment. Rendering requires an NVIDIA CUDA GPU and a compatible build environment. Select dependencies for your own operating system, GPU and driver using the instructions below.
If your computer does not support CUDA, you can use [Google Colab](https://colab.research.google.com/) with a GPU runtime.

---

## Table of Contents

- [Embodied AI Homework 1 — 3DGS](#embodied-ai-homework-1--3dgs)
  - [Table of Contents](#table-of-contents)
  - [0. Setting up the environment](#0-setting-up-the-environment)
  - [1. Preparation](#1-preparation)
    - [Dataset and COLMAP](#dataset-and-colmap)
    - [Point cloud initialization](#point-cloud-initialization)
  - [2. Building Gaussian primitives](#2-building-gaussian-primitives)
  - [3. Better rendering](#3-better-rendering)
  - [4. About submission](#4-about-submission)
  - [References](#references)

---

## 0. Setting up the environment

Use an independent **venv or Conda environment**. Use **64-bit CPython 3.10–3.13**, with a compatible PyTorch build and dependency wheels for your platform. The pinned NumPy version does not provide Python 3.14 wheels; a newer Python interpreter may trigger an unverified source build.

Prepare these prerequisites:

- An NVIDIA GPU and a working driver. `nvidia-smi` should detect the GPU. Choose a CUDA-enabled PyTorch build that supports your GPU and driver; this assignment does not include a CPU rasterizer.
- A local **CUDA Toolkit**, including `nvcc`, compatible with that PyTorch build. Prefer a Toolkit matching the CUDA major/minor version reported by `torch.version.cuda`. A PyTorch wheel's bundled CUDA runtime does not include the compiler needed to build this assignment's extension. Put the Toolkit tools on `PATH`; set `CUDA_HOME` to its installation directory if automatic detection fails.
- A C++ compiler supported by your Toolkit, such as a compatible GCC/G++ on Linux or MSVC C++ build tools on Windows. On Windows, use a developer shell that can find the compiler. Select the intended Toolkit and compiler if several are installed.

The CUDA version displayed by `nvidia-smi` describes driver support. `nvcc --version` describes the installed Toolkit, and `torch.version.cuda` describes the CUDA version used to build PyTorch. Check these components separately.

Run the following steps from the assignment root:

1. Create and activate your environment. For venv, use your selected Python interpreter:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   In Windows PowerShell, use `.\.venv\Scripts\Activate.ps1` to activate it. If using Conda, activate your chosen Conda environment instead.

2. Install **CUDA-enabled PyTorch and matching torchvision** using the official installation selector for your operating system and selected CUDA build.

   **Official PyTorch website:** <https://pytorch.org/>  
   **Official installation selector:** <https://pytorch.org/get-started/locally/>

   Choose builds compatible with the NumPy 2 dependency in `requirements.txt`. PyTorch and torchvision are intentionally installed separately from that file. Check them in the same active environment:

   ```bash
   python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
   ```

   CUDA must be available before rendering.

3. Install the remaining dependencies and compile the rasterization extension. In a Bash shell, use:

   ```bash
   bash scripts/setup_environment.sh
   ```

   This helper uses the **currently active environment** and its preinstalled PyTorch. It checks CUDA prerequisites, installs dependencies and builds the extension with your selected compiler. You must prepare the environment, driver and toolchain yourself.

   Alternatively, including on a platform without Bash, run:

   ```bash
   python -m pip install "setuptools>=64" wheel
   python -m pip install -r requirements.txt
   python scripts/check_environment.py --check-runtime
   python -m pip install --no-build-isolation --no-deps --force-reinstall ./submodules/diff-gaussian-rasterization
   ```

   These commands build and install the extension in the active environment. Rebuild it after changing its CUDA source, PyTorch installation or toolchain. Compatibility must be checked in your environment; not every platform combination has been tested.

   Adjust compilation parallelism to available memory with `MAX_JOBS`, for example `export MAX_JOBS=4` in Bash or `$env:MAX_JOBS=4` in PowerShell. The Bash helper defaults to four jobs.

**Quick check (coding, 5 pts):**

```bash
python render.py
```

After a successful environment setup, the intentionally unfinished starter functions should produce **incorrectly rendered** images in `renders/`. Include **one** baseline image in your report to receive 5 points. These images are the starting point for the implementation tasks below. **Save a separate copy of your baseline images before editing the TODOs:** later runs overwrite the same files in `renders/`.

---

## 1. Preparation

Complete the environment setup and extension build in Section 0 before continuing.

### Dataset and COLMAP

We provide a simple scene **“The Fruity Ice-cream.”**

- The images are in `assets/fruit.zip`. Extract them and use **COLMAP** to generate a **sparse point cloud**. Dense reconstruction is not required.
- You may install the standalone COLMAP application and use its GUI or CLI. Alternatively, you may use the **PyCOLMAP** Python bindings and write your own reconstruction script. Installing `pycolmap` does **not** install the standalone `colmap` command. PyCOLMAP is optional and is not installed by the base requirements. If you choose this route and a wheel is available for your platform, install it separately:

  ```bash
  python -m pip install pycolmap==4.2.1
  ```

  Otherwise, use the standalone COLMAP application or follow the official instructions to build the bindings. Refer to the official COLMAP documentation for installation and usage.

The dataset contains images of different dimensions. Allow camera grouping appropriate to those dimensions rather than forcing every image to share one camera. Check the logs for skipped images and report the numbers imported and registered.

**Task (writing && coding, 10 pts):** Watch the logs and **identify the stages** involved in the reconstruction pipeline. Include your observations in your report.

### Point cloud initialization

Since **bundle adjustment** can be time-consuming, we provide a precomputed point cloud:

- `assets/points3D.ply`

**Task (coding, 10 pts):** Visualize this point cloud, which serves as initialization of the Gaussians, with **MeshLab**, **Open3D**, or another tool you prefer. Include a screenshot in your report.

> You are **not required to train a 3DGS model**. You may explore training using the official implementation if interested. Instead, we provide optimized Gaussians in `assets/gs_cloud.ply` and an inference codebase with core sections marked `TODO` for you to implement.

---

## 2. Building Gaussian primitives

In this section, you will construct **elliptic Gaussian** primitives and **splat** them into screen space.

- We denote a 3D Gaussian by its mean and covariance.

**(writing && coding, 15 pts)**

- Derive the **covariance matrix** from a **scaling matrix** and a **rotation matrix**.
- Complete the `TODO` in `get_covariance` in `gaussian_model.py`.

**(writing, 5 pts)**

- Suppose a **viewing transformation** $\mathbf{T}_v$ is applied to a Gaussian with mean $\mu$ and covariance $\Sigma$. Write down the **transformed Gaussian** in **camera space**.

**(writing, 10 pts)**

- Suppose a **projection transformation** is applied. Although the mapping is **nonlinear**, approximate it **locally** at the mean by a **linear transformation** (its Jacobian). Derive the expression of the Jacobian and the **new covariance** of the transformed Gaussian.

**(coding, 15 pts)**

- By **sequentially applying** the viewing and projection transformations, you will **splat the 3D Gaussian** into a **2D screen-space Gaussian**.
- Complete the `TODO` in `computeCov2D` in `submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu`.

**Rebuild the CUDA extension after changes:**

```bash
cd submodules/diff-gaussian-rasterization
python setup.py clean --all
python -m pip install --no-build-isolation --no-deps --force-reinstall .
cd ../..
```

**(writing, 10 pts)**

- Given a **screen-space Gaussian covariance** $\Sigma_{2D}$, find a **square bounding-box side length** that covers the **99% confidence** region of the **joint two-dimensional distribution**. This is a written derivation; you do not need to change the rasterizer's existing truncation radius for this question.

> After implementing the above, run `python render.py` and inspect the images generated in `renders/`.

---

## 3. Better rendering

**(coding, 10 pts) First-degree spherical harmonics (SH)**

- The starter code supports only **0th-order SH**.
- Implement **1st-degree SH** by completing the `TODO` in `eval_sh` in `utils/sh_utils.py`. Refer to the table of spherical harmonics in the **References**. Match the basis and coefficient conventions used by the provided code and model.
- Evaluate rendering performance **before vs. after** and include a comparison in your report. Use `python render.py --sh_degree 0` and `python render.py --sh_degree 1`, saving each run separately. Compare image quality and rendering time over all five supplied cameras. The corresponding reference images are `assets/gt/0.png` through `4.png`, in the same order as `assets/cameras/camera_0.npy` through `camera_4.npy`.

**(coding, 10 pts) Post-hoc pruning**

- Implement **opacity-based pruning** by completing the `TODO` in `prune_points` in `gaussian_model.py`.
- Experiment with various **opacity thresholds** and report the **maximum pruning ratio** that **does not** introduce visible artifacts. Include comparisons supporting your choice. Report the largest acceptable ratio **among your tested thresholds**, using all five provided views and stating your visual acceptance criteria; no proof of a global optimum over unseen views is required.

**(bonus, 20 pts)**

- Propose and evaluate an **improved post-hoc pruning** strategy. Explain the method and show its performance.

The ten required items total **100 points**, with **20 additional bonus points**.

---

## 4. About submission

- **What to submit:** Your **code** and your own **written report**, packed as a `.zip`. The report must contain the requested written derivations, baseline image, reconstruction observations, point cloud screenshot and rendering/pruning comparisons. Include your bonus method and evaluation if attempted.
- **Where to submit:** **Web Learning**.
- **Deadline:** **Nov 5, 2026, 23:59 Beijing time, UTC+08:00**.

---

## References

- COLMAP documentation: <https://colmap.github.io/>
- Spherical harmonics (table for $\ell=1,2$): <https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#%E2%84%93_=_1_2>
