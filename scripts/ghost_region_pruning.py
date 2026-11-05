"""Region-level pruning helpers for translucent ghost Gaussians.

Images are (H, W, 3) float tensors in [0, 1] (the layout of render.py output after permute).
Gaussian tensors stay on the device of the source model.
"""

import math
import numpy as np
import scipy.sparse as scipy_sparse
import scipy.sparse.csgraph as scipy_csgraph
import scipy.spatial as scipy_spatial
import torch
import torch.nn as nn
import gaussian_model

NEAR_PLANE_DEPTH = 0.2
SH_C0 = 0.28209479177387814


def translucent_mask(opacity, max_opacity):
    return opacity <= max_opacity


def base_colors(model):
    """RGB in [0, 1] from each Gaussian's DC spherical-harmonic term (view-independent part of the color)."""
    return (model.get_features[:, 0, :].detach() * SH_C0 + 0.5).clamp(0.0, 1.0)


def error_map(render_hwc, reference_hwc):
    """Largest per-channel absolute difference, the quantity the flagged-pixel test thresholds."""
    return (render_hwc - reference_hwc).abs().max(dim=-1).values


def flagged_pixel_mask(render_hwc, reference_hwc, diff_threshold):
    return error_map(render_hwc, reference_hwc) >= diff_threshold


def dilate_mask(mask_hw, radius_px):
    if radius_px == 0:
        return mask_hw
    kernel_size = 2 * radius_px + 1
    dilated = torch.nn.functional.max_pool2d(
        mask_hw[None, None].float(), kernel_size, stride=1, padding=radius_px
    )
    return dilated[0, 0] > 0.5


def project_centers(camera, centers_xyz):
    pixels_xy, inside, _ = project_centers_full(camera, centers_xyz)
    return pixels_xy, inside


def project_centers_full(camera, centers_xyz):
    """Project centers to pixel coordinates. Returns (pixels_xy, inside, depth)."""
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
    return torch.stack((pixel_x, pixel_y), dim=1), inside, view_points[:, 2]


def screen_radii(camera, largest_scale, depth):
    """Approximate screen-space radius in pixels of each Gaussian's splat."""
    focal_pixels = 0.5 * float(camera.image_width) / math.tan(0.5 * float(camera.FoVx))
    return largest_scale * focal_pixels / depth.clamp_min(NEAR_PLANE_DEPTH)


def integral_image(values_hw):
    """Summed-area table, padded so box_sums can index inclusively."""
    return torch.nn.functional.pad(values_hw, (1, 0, 1, 0)).cumsum(dim=0).cumsum(dim=1)


def box_sums(integral_hw, rows, columns, radius):
    """Sum of the (2 * radius + 1) square around each pixel, clipped to the image."""
    height = integral_hw.shape[0] - 1
    width = integral_hw.shape[1] - 1
    radius = radius.clamp_min(0)
    top = (rows - radius).clamp(0, height)
    bottom = (rows + radius + 1).clamp(0, height)
    left = (columns - radius).clamp(0, width)
    right = (columns + radius + 1).clamp(0, width)
    return (integral_hw[bottom, right] - integral_hw[top, right]
            - integral_hw[bottom, left] + integral_hw[top, left])


def scatter_boxes(values, rows, columns, radius, height, width):
    """Per-pixel sum of `values` over the boxes centred at (rows, columns).

    Uses a 2D difference array, so the cost is one scatter per Gaussian instead of
    one write per covered pixel.
    """
    buffer = torch.zeros((height + 1, width + 1), dtype=values.dtype, device=values.device)
    radius = radius.clamp_min(0)
    top = (rows - radius).clamp(0, height)
    bottom = (rows + radius + 1).clamp(0, height)
    left = (columns - radius).clamp(0, width)
    right = (columns + radius + 1).clamp(0, width)
    corner_rows = torch.cat((top, top, bottom, bottom))
    corner_columns = torch.cat((left, right, left, right))
    corner_values = torch.cat((values, -values, -values, values))
    buffer.index_put_((corner_rows, corner_columns), corner_values, accumulate=True)
    return buffer.cumsum(dim=0).cumsum(dim=1)[:height, :width]


def evidence_mask(pixels_xy, inside, flagged_hw):
    height, width = flagged_hw.shape
    columns = torch.round(pixels_xy[:, 0]).long().clamp(0, width - 1)
    rows = torch.round(pixels_xy[:, 1]).long().clamp(0, height - 1)
    return flagged_hw[rows, columns] & inside


def build_feature_clusters(xyz_np, rgb_np, opacity_np, candidate_mask_np, evidence_np,
                           radius, color_tolerance, opacity_tolerance):
    """Group translucent Gaussians that are close in position, color, and opacity.

    xyz_np (N, 3), rgb_np (N, 3), opacity_np (N,): per-Gaussian attributes.
    candidate_mask_np (N,): translucent Gaussians. evidence_np (V, N) bool: the
    center falls in a flagged region of view v.
    Two candidates link when their centers are within `radius`, every RGB channel
    differs by at most `color_tolerance`, and their opacities differ by at most
    `opacity_tolerance`. A cluster is a connected component of that graph that
    contains at least one Gaussian with evidence. Returns (clusters, stats).
    """
    candidate_indices = np.flatnonzero(candidate_mask_np)
    candidate_count = int(candidate_indices.size)
    candidate_xyz = xyz_np[candidate_indices]
    candidate_rgb = rgb_np[candidate_indices]
    candidate_opacity = opacity_np[candidate_indices]

    pairs = scipy_spatial.cKDTree(candidate_xyz).query_pairs(r=radius, output_type='ndarray')
    pairs_within_radius = len(pairs)
    color_gap = np.abs(candidate_rgb[pairs[:, 0]] - candidate_rgb[pairs[:, 1]]).max(axis=1)
    pairs = pairs[color_gap <= color_tolerance]
    pairs_within_color = len(pairs)
    opacity_gap = np.abs(candidate_opacity[pairs[:, 0]] - candidate_opacity[pairs[:, 1]])
    pairs = pairs[opacity_gap <= opacity_tolerance]
    pairs_within_opacity = len(pairs)

    linked = scipy_sparse.coo_matrix(
        (np.ones(pairs_within_opacity, dtype=np.int8), (pairs[:, 0], pairs[:, 1])),
        shape=(candidate_count, candidate_count),
    )
    component_count, labels = scipy_csgraph.connected_components(linked, directed=False)

    evidence_per_candidate = evidence_np[:, candidate_indices].any(axis=0)
    component_has_evidence = np.zeros(component_count, dtype=bool)
    np.logical_or.at(component_has_evidence, labels, evidence_per_candidate)

    nodes = np.flatnonzero(component_has_evidence[labels])
    nodes = nodes[np.argsort(labels[nodes], kind='stable')]
    boundaries = np.flatnonzero(np.diff(labels[nodes])) + 1
    groups = np.split(nodes, boundaries) if nodes.size else []

    clusters = []
    for local in groups:
        members = candidate_indices[local]
        member_evidence = evidence_np[:, members]
        member_rgb = rgb_np[members]
        member_opacity = opacity_np[members]
        clusters.append({
            'id': len(clusters),
            'member_indices': members,
            'member_count': int(members.size),
            'evidence_gaussians': int(evidence_per_candidate[local].sum()),
            'evidence_hits': int(member_evidence.sum()),
            'views_with_evidence': int(member_evidence.any(axis=1).sum()),
            'bbox_min': xyz_np[members].min(axis=0),
            'bbox_max': xyz_np[members].max(axis=0),
            'rgb_mean': member_rgb.mean(axis=0),
            'rgb_spread': member_rgb.max(axis=0) - member_rgb.min(axis=0),
            'opacity_mean': float(member_opacity.mean()),
            'opacity_spread': float(member_opacity.max() - member_opacity.min()),
        })

    stats = {
        'candidates': candidate_count,
        'pairs_within_radius': pairs_within_radius,
        'pairs_within_color': pairs_within_color,
        'pairs_within_opacity': pairs_within_opacity,
        'components': int(component_count),
        'components_with_evidence': int(component_has_evidence.sum()),
    }
    return clusters, stats


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
