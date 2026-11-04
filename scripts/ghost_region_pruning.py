"""Region-level pruning helpers for translucent ghost Gaussians.

Images are (H, W, 3) float tensors in [0, 1] (the layout of render.py output after permute).
Gaussian tensors stay on the device of the source model.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import gaussian_model

NEAR_PLANE_DEPTH = 0.2
NEIGHBOR_OFFSETS = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
]


def translucent_mask(opacity, max_opacity):
    return opacity <= max_opacity


def flagged_pixel_mask(render_hwc, reference_hwc, diff_threshold):
    largest_channel_difference = (render_hwc - reference_hwc).abs().max(dim=-1).values
    return largest_channel_difference >= diff_threshold


def dilate_mask(mask_hw, radius_px):
    if radius_px == 0:
        return mask_hw
    kernel_size = 2 * radius_px + 1
    dilated = torch.nn.functional.max_pool2d(
        mask_hw[None, None].float(), kernel_size, stride=1, padding=radius_px
    )
    return dilated[0, 0] > 0.5


def project_centers(camera, centers_xyz):
    ones = torch.ones((centers_xyz.shape[0], 1), dtype=centers_xyz.dtype, device=centers_xyz.device)
    homogeneous = torch.cat((centers_xyz, ones), dim=1)
    view_points = homogeneous @ camera.world_view_transform
    clip = homogeneous @ camera.full_proj_transform
    inverse_w = 1.0 / (clip[:, 3] + 1e-7)
    width = float(camera.image_width)
    height = float(camera.image_height)
    pixel_x = ((clip[:, 0] * inverse_w + 1.0) * width - 1.0) * 0.5
    pixel_y = ((clip[:, 1] * inverse_w + 1.0) * height - 1.0) * 0.5
    inside = (
        (view_points[:, 2] > NEAR_PLANE_DEPTH)
        & (pixel_x >= -0.5) & (pixel_x < width - 0.5)
        & (pixel_y >= -0.5) & (pixel_y < height - 0.5)
    )
    return torch.stack((pixel_x, pixel_y), dim=1), inside


def evidence_mask(pixels_xy, inside, flagged_hw):
    height, width = flagged_hw.shape
    columns = torch.round(pixels_xy[:, 0]).long().clamp(0, width - 1)
    rows = torch.round(pixels_xy[:, 1]).long().clamp(0, height - 1)
    return flagged_hw[rows, columns] & inside


def build_voxel_clusters(xyz_np, candidate_mask_np, evidence_np, voxel_size):
    """Group candidate Gaussians into 26-connected clusters of occupied voxels.

    xyz_np: (N, 3) centers; candidate_mask_np: (N,) translucent Gaussians;
    evidence_np: (V, N) bool, Gaussian center falls in a flagged region of view v.
    Only voxels containing at least one evidence Gaussian seed clusters; every
    candidate Gaussian in a seeded voxel joins that voxel's cluster.
    """
    candidate_indices = np.flatnonzero(candidate_mask_np)
    candidate_xyz = xyz_np[candidate_indices]
    origin = candidate_xyz.min(axis=0)
    voxel_coords = np.floor((candidate_xyz - origin) / voxel_size).astype(np.int64)
    unique_voxels, voxel_of_candidate = np.unique(voxel_coords, axis=0, return_inverse=True)
    voxel_of_candidate = voxel_of_candidate.reshape(-1)

    evidence_per_candidate = evidence_np[:, candidate_indices].any(axis=0)
    voxel_has_evidence = np.zeros(len(unique_voxels), dtype=bool)
    np.logical_or.at(voxel_has_evidence, voxel_of_candidate, evidence_per_candidate)

    evidence_voxels = [int(voxel) for voxel in np.flatnonzero(voxel_has_evidence)]
    voxel_id_by_coord = {tuple(unique_voxels[voxel].tolist()): voxel for voxel in evidence_voxels}
    parent = {voxel: voxel for voxel in evidence_voxels}

    def find_root(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for voxel in evidence_voxels:
        coord = unique_voxels[voxel].tolist()
        for dx, dy, dz in NEIGHBOR_OFFSETS:
            neighbor = voxel_id_by_coord.get((coord[0] + dx, coord[1] + dy, coord[2] + dz))
            if neighbor is None:
                continue
            root_a, root_b = find_root(voxel), find_root(neighbor)
            if root_a != root_b:
                parent[root_b] = root_a

    root_of_voxel = np.full(len(unique_voxels), -1, dtype=np.int64)
    for voxel in evidence_voxels:
        root_of_voxel[voxel] = find_root(voxel)

    root_of_candidate = root_of_voxel[voxel_of_candidate]
    clusters = []
    for root in np.unique(root_of_candidate[root_of_candidate >= 0]):
        local = np.flatnonzero(root_of_candidate == root)
        members = candidate_indices[local]
        member_evidence = evidence_np[:, members]
        clusters.append({
            'id': len(clusters),
            'member_indices': members,
            'member_count': int(members.size),
            'evidence_gaussians': int(evidence_per_candidate[local].sum()),
            'evidence_hits': int(member_evidence.sum()),
            'views_with_evidence': int(member_evidence.any(axis=1).sum()),
            'bbox_min': xyz_np[members].min(axis=0),
            'bbox_max': xyz_np[members].max(axis=0),
        })
    return clusters


def select_clusters(clusters, min_views, max_members, top_k):
    eligible = [
        cluster for cluster in clusters
        if cluster['views_with_evidence'] >= min_views and cluster['member_count'] <= max_members
    ]
    ranked = sorted(eligible, key=lambda cluster: cluster['evidence_hits'], reverse=True)
    return ranked[:top_k]


def keep_mask_without(count, removed_indices, device):
    keep = torch.ones(count, dtype=torch.bool, device=device)
    keep[removed_indices] = False
    return keep


def subset_model(source, keep_mask):
    pruned = gaussian_model.GaussianModel(source.active_sh_degree)
    pruned._xyz = nn.Parameter(source._xyz.detach()[keep_mask])
    pruned._features_dc = nn.Parameter(source._features_dc.detach()[keep_mask])
    pruned._features_rest = nn.Parameter(source._features_rest.detach()[keep_mask])
    pruned._opacity = nn.Parameter(source._opacity.detach()[keep_mask])
    pruned._scaling = nn.Parameter(source._scaling.detach()[keep_mask])
    pruned._rotation = nn.Parameter(source._rotation.detach()[keep_mask])
    return pruned


def lowest_opacity_indices(opacity, count):
    return torch.argsort(opacity)[:count]


def image_metrics(render_hwc, reference_hwc, diff_threshold):
    difference = render_hwc - reference_hwc
    mse = float(difference.square().mean())
    flagged = flagged_pixel_mask(render_hwc, reference_hwc, diff_threshold)
    return {
        'mae': float(difference.abs().mean()),
        'psnr_db': float('inf') if mse == 0 else 10.0 * math.log10(1.0 / mse),
        'flagged_pixels': int(flagged.sum()),
    }


def cluster_is_accepted(before, after, max_mae_regression):
    """before/after: per-selection-view metric dicts from image_metrics."""
    flagged_before = sum(metric['flagged_pixels'] for metric in before)
    flagged_after = sum(metric['flagged_pixels'] for metric in after)
    mae_within_tolerance = all(
        new['mae'] - old['mae'] <= max_mae_regression for old, new in zip(before, after)
    )
    return flagged_after < flagged_before and mae_within_tolerance


def mark_pixels(image_hwc, mask_hw, color=(255, 0, 0)):
    image = (image_hwc.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().numpy()
    image[mask_hw.cpu().numpy()] = color
    return image
