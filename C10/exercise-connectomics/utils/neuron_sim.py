"""
neuron_sim.py
-------------
Utilities for generating synthetic fluorescence microscopy images
of neurons with soma, axons, dendrites, and synaptic connections.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import random


@dataclass
class Synapse:
    pre_id: int          # index of pre-synaptic neuron
    post_id: int         # index of post-synaptic neuron
    location: Tuple[float, float]   # (row, col) in image coords
    strength: float = 1.0


@dataclass
class Neuron:
    neuron_id: int
    soma_center: Tuple[float, float]   # (row, col)
    soma_radius: float
    axon_endpoints: List[Tuple[float, float]] = field(default_factory=list)
    dendrite_endpoints: List[Tuple[float, float]] = field(default_factory=list)
    color_channel: int = 0   # 0=red, 1=green, 2=blue (for multi-channel sim)


def _draw_gaussian_blob(canvas: np.ndarray, center: Tuple[float, float],
                         radius: float, intensity: float = 1.0) -> None:
    """Draw a soft gaussian blob on canvas (in-place)."""
    r, c = int(center[0]), int(center[1])
    half = int(radius * 3)
    H, W = canvas.shape
    for dr in range(-half, half + 1):
        for dc in range(-half, half + 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W:
                dist2 = dr ** 2 + dc ** 2
                canvas[rr, cc] += intensity * np.exp(-dist2 / (2 * radius ** 2))


def _draw_line(canvas: np.ndarray, p0: Tuple[float, float],
               p1: Tuple[float, float], width: float = 1.5,
               intensity: float = 0.6) -> None:
    """Draw a soft anti-aliased line segment on canvas (in-place)."""
    r0, c0 = p0
    r1, c1 = p1
    length = np.hypot(r1 - r0, c1 - c0)
    n_steps = max(int(length * 2), 2)
    H, W = canvas.shape
    for i in range(n_steps + 1):
        t = i / n_steps
        r = r0 + t * (r1 - r0)
        c = c0 + t * (c1 - c0)
        ri, ci = int(r), int(c)
        for dr in range(-int(width * 2), int(width * 2) + 1):
            for dc in range(-int(width * 2), int(width * 2) + 1):
                rr, cc = ri + dr, ci + dc
                if 0 <= rr < H and 0 <= cc < W:
                    dist = np.sqrt(dr ** 2 + dc ** 2)
                    canvas[rr, cc] += intensity * np.exp(-dist ** 2 / (2 * width ** 2))


def _draw_branching_axon(canvas: np.ndarray, start: Tuple[float, float],
                          direction: float, length: float, depth: int,
                          width: float, intensity: float,
                          rng: np.random.Generator) -> List[Tuple[float, float]]:
    """
    Recursively draw a branching axon. Returns list of tip positions.
    direction: angle in radians
    """
    H, W = canvas.shape
    end_r = start[0] + length * np.sin(direction)
    end_c = start[1] + length * np.cos(direction)
    end_r = np.clip(end_r, 5, H - 5)
    end_c = np.clip(end_c, 5, W - 5)
    end = (end_r, end_c)
    _draw_line(canvas, start, end, width=width, intensity=intensity)

    if depth == 0:
        return [end]

    tips = []
    n_branches = rng.integers(1, 3)
    for _ in range(n_branches):
        branch_angle = direction + rng.uniform(-np.pi / 4, np.pi / 4)
        branch_length = length * rng.uniform(0.5, 0.8)
        tips.extend(_draw_branching_axon(
            canvas, end, branch_angle, branch_length,
            depth - 1, width * 0.7, intensity * 0.8, rng
        ))
    return tips


def generate_neuron_scene(
    image_size: Tuple[int, int] = (512, 512),
    n_neurons: int = 8,
    seed: int = 42,
    noise_level: float = 0.05,
    background_level: float = 0.02,
) -> Tuple[np.ndarray, List[Neuron], List[Synapse]]:
    """
    Generate a synthetic fluorescence microscopy image of neurons.

    Parameters
    ----------
    image_size : (H, W)
    n_neurons  : number of neurons to place
    seed       : random seed for reproducibility
    noise_level: Gaussian noise sigma (fraction of max intensity)
    background_level: background autofluorescence level

    Returns
    -------
    image   : float32 array of shape (H, W), values in [0, 1]
    neurons : list of Neuron objects
    synapses: list of Synapse objects (ground-truth connections)
    """
    rng = np.random.default_rng(seed)
    H, W = image_size
    canvas = np.full((H, W), background_level, dtype=np.float32)

    # Place soma centers with minimum separation
    soma_centers = []
    min_sep = min(H, W) / (n_neurons ** 0.5 + 1)
    attempts = 0
    while len(soma_centers) < n_neurons and attempts < 10000:
        r = rng.uniform(60, H - 60)
        c = rng.uniform(60, W - 60)
        if all(np.hypot(r - sr, c - sc) > min_sep for sr, sc in soma_centers):
            soma_centers.append((r, c))
        attempts += 1

    neurons = []
    axon_tips_per_neuron = []

    for i, center in enumerate(soma_centers):
        soma_radius = rng.uniform(10, 18)
        neuron = Neuron(
            neuron_id=i,
            soma_center=center,
            soma_radius=soma_radius,
        )

        # Draw soma
        _draw_gaussian_blob(canvas, center, soma_radius, intensity=1.0)

        # Draw axon trunk + branches
        axon_direction = rng.uniform(0, 2 * np.pi)
        axon_length = rng.uniform(60, 120)
        tips = _draw_branching_axon(
            canvas, center, axon_direction, axon_length,
            depth=2, width=2.0, intensity=0.7, rng=rng
        )
        neuron.axon_endpoints = tips
        axon_tips_per_neuron.append(tips)

        # Draw 2-3 short dendrites
        for _ in range(rng.integers(2, 4)):
            dend_dir = rng.uniform(0, 2 * np.pi)
            dend_len = rng.uniform(20, 50)
            dend_end_r = center[0] + dend_len * np.sin(dend_dir)
            dend_end_c = center[1] + dend_len * np.cos(dend_dir)
            dend_end = (
                np.clip(dend_end_r, 5, H - 5),
                np.clip(dend_end_c, 5, W - 5)
            )
            _draw_line(canvas, center, dend_end, width=1.2, intensity=0.5)
            neuron.dendrite_endpoints.append(dend_end)

        neurons.append(neuron)

    # Build synapses: connect axon tips to nearby soma of different neurons
    synapses = []
    synapse_radius = min(H, W) / 8.0
    for pre_id, tips in enumerate(axon_tips_per_neuron):
        for tip in tips:
            best_post = None
            best_dist = synapse_radius
            for post_id, neuron in enumerate(neurons):
                if post_id == pre_id:
                    continue
                dist = np.hypot(tip[0] - neuron.soma_center[0],
                                tip[1] - neuron.soma_center[1])
                if dist < best_dist:
                    best_dist = dist
                    best_post = post_id
            if best_post is not None:
                strength = float(np.exp(-best_dist / synapse_radius))
                synapses.append(Synapse(
                    pre_id=pre_id,
                    post_id=best_post,
                    location=tip,
                    strength=strength
                ))
                # Draw synapse marker (bright dot)
                _draw_gaussian_blob(canvas, tip, radius=4.0, intensity=1.2)

    # Add Poisson-like shot noise + Gaussian blur (PSF)
    canvas = gaussian_filter(canvas, sigma=1.5)
    noise = rng.normal(0, noise_level, (H, W)).astype(np.float32)
    canvas = np.clip(canvas + noise, 0, None)

    # Normalize to [0, 1]
    canvas = canvas / (canvas.max() + 1e-8)

    return canvas.astype(np.float32), neurons, synapses


def make_ground_truth_masks(
    image_size: Tuple[int, int],
    neurons: List[Neuron],
    synapses: List[Synapse]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build binary ground-truth masks for supervised training.

    Returns
    -------
    soma_mask    : uint8 array (H, W), 1 inside soma regions
    synapse_mask : uint8 array (H, W), 1 at synapse locations
    """
    H, W = image_size
    soma_mask = np.zeros((H, W), dtype=np.uint8)
    synapse_mask = np.zeros((H, W), dtype=np.uint8)

    for neuron in neurons:
        r, c = neuron.soma_center
        ri, ci = int(r), int(c)
        rad = int(neuron.soma_radius * 1.2)
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                if dr ** 2 + dc ** 2 <= rad ** 2:
                    rr, cc = ri + dr, ci + dc
                    if 0 <= rr < H and 0 <= cc < W:
                        soma_mask[rr, cc] = 1

    for syn in synapses:
        r, c = int(syn.location[0]), int(syn.location[1])
        for dr in range(-5, 6):
            for dc in range(-5, 6):
                if dr ** 2 + dc ** 2 <= 25:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W:
                        synapse_mask[rr, cc] = 1

    return soma_mask, synapse_mask
