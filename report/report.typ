
= Embodied AI \ Homework 1

\

Franz Jonas Ullmann \ 林阳 \ 2026400694

= Task 0: Environment setup and baseline rendering

== Environment and result

The notebook #link("task0_colab.ipynb")[task0_colab.ipynb] ran through VS Code using a Google Colab GPU runtime. The recorded environment was:

- GPU: Tesla T4.
- Python: 3.13.16.
- PyTorch: 2.11.0+cu130; torchvision: 0.26.0+cu130.
- PyTorch CUDA and CUDA Toolkit: 13.0; Toolkit path: `/usr/local/cuda`.

The GPU and runtime interoperability checks passed. After the build issue described below was fixed, the CUDA rasterizer built and imported successfully. The starter implementation rendered five camera views, which the notebook preserved separately from `renders/`. These are baseline results from the unfinished implementation, not evidence that the later TODOs are complete.

== What did not work

The connection to the Colab GPU was working, but the rasterizer extension initially failed to compile. The CUDA compiler reported that `std::uintptr_t`, `uint32_t`, and `uint64_t` were undefined in `rasterizer_impl.h`. The initial pip output only reported a failed wheel build and "No available output", so it did not reveal the underlying compiler error. A verbose rebuild exposed the missing integer type declarations.

Adding the standard headers `<cstddef>`, `<cstdint>`, and `<stdint.h>` to the rasterizer header resolved the observed build errors. The notebook includes a cell that applies the fix to the temporary Colab checkout; a subsequent build and render succeeded. The build output is now streamed and pip is run verbosely to make future compiler errors visible.

Colab clones or pulls the GitHub repository rather than using uncommitted local files. Therefore local changes must be committed and pushed before a fresh runtime can pick them up. The header repair cell modifies the temporary checkout, so a fresh runtime should be started after pushing the permanent fix instead of pulling over that modification.

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
2. *Feature matching and geometric verification:* COLMAP matched features between images and filtered matches that were inconsistent with the estimated geometry.
3. *Incremental mapping:* COLMAP initialized a reconstruction and registered images successively, triangulating 3D points from the verified matches.
4. *Retriangulation and global bundle adjustment:* COLMAP refined the points, camera poses, and camera parameters as more images were registered.

The mapper registered 219 images. It emitted several linear-solver warnings during optimization, but finished by keeping a successful reconstruction. The log reports about 0.27 minutes for feature extraction, 3.94 minutes for feature matching and geometric verification, and 11.90 minutes for mapping.

For camera initialization, I selected *Parameters from EXIF* in the GUI. The log nevertheless assigns a different `SIMPLE_RADIAL` camera ID to each image, including images 001 and 002, which have the same 1399 x 1439 resolution. Their reported focal length is 1726.80 pixels, consistent with COLMAP's default initialization for that image size rather than a shared camera calibration. I kept this reconstruction because the camera groups are unknown; consequently, its camera models were not shared across images.

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
