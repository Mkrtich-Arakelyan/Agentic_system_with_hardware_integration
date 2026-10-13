"""A random-matrix optical bench; intensities are in relative simulation units."""

import numpy as np


SEARCH_SHOTS = 3
VALIDATION_SHOTS = 5
DENOMINATOR_FLOOR = 1e-12
BACKGROUND_REGIONS = ("top_left", "top_right", "bottom_left", "bottom_right")


def _binary_array(value, shape):
    """Check the grid shape and binary values before using a DMD pattern."""
    array = np.asarray(value)
    if array.shape != shape or not np.all((array == 0) | (array == 1)):
        raise ValueError(f"Expected a binary pattern with shape {shape}.")
    return array.astype(np.uint8)


def _image_statistics(image, masks) -> dict:
    """Divide the focus mean by the equally weighted mean of four region means."""
    focus_mean = float(image[masks["focus"]].mean())
    background_means = {
        name: float(image[masks[name]].mean()) for name in BACKGROUND_REGIONS
    }
    reference = float(np.mean(list(background_means.values())))
    return {
        "enhancement": focus_mean / max(reference, DENOMINATOR_FLOOR),
        "focus_mean": focus_mean,
        "background_means": background_means,
        "background_reference": reference,
    }


class SimulatedBench:
    """Simulate a binary amplitude DMD, scattering, and a noisy camera."""

    def __init__(self, dmd_shape=(16, 16), camera_shape=(32, 32),
                 transmission_seed=42, noise_seed=43, noise_std=0.01,
                 target_position=None):
        for shape in (dmd_shape, camera_shape):
            if len(shape) != 2 or any(not isinstance(n, int) or n <= 0 for n in shape):
                raise ValueError("DMD and camera shapes need two positive integers.")
        if not np.isfinite(noise_std) or noise_std < 0:
            raise ValueError("noise_std must be finite and nonnegative.")
        if target_position is None:
            target_position = (camera_shape[0] // 2, camera_shape[1] // 2)
        if len(target_position) != 2 or any(
            not isinstance(p, int) or not 0 <= p < n
            for p, n in zip(target_position, camera_shape)
        ):
            raise ValueError("The target must be a (row, column) inside the camera.")

        self.dmd_shape = dmd_shape
        self.camera_shape = camera_shape
        self.target_position = target_position
        self.transmission_seed = transmission_seed
        self.noise_seed = noise_seed
        self.noise_std = noise_std
        matrix_rng = np.random.default_rng(transmission_seed)
        self._noise_rng = np.random.default_rng(noise_seed)
        dmd_pixels = int(np.prod(dmd_shape))
        matrix_shape = (int(np.prod(camera_shape)), dmd_pixels)
        # Generate scattering once: changing macropixels must not change the bench.
        self._T = (
            matrix_rng.standard_normal(matrix_shape)
            + 1j * matrix_rng.standard_normal(matrix_shape)
        ) / np.sqrt(2 * dmd_pixels)
        self._T.setflags(write=False)

    def capture(self, pattern):
        """Return one noisy camera image of a nonzero binary DMD pattern."""
        pattern = _binary_array(pattern, self.dmd_shape)
        if not pattern.any():
            raise ValueError("The DMD pattern cannot be all off.")
        intensity = (np.abs(self._T @ pattern.ravel()) ** 2).reshape(self.camera_shape)
        # Reuse the advancing RNG so each capture gets fresh measurement noise.
        noise = self._noise_rng.normal(0, self.noise_std, self.camera_shape)
        return np.maximum(intensity + noise, 0)

    def macropixel_shape(self, macropixel_size):
        """Return the smaller control-grid shape for a valid block size."""
        size = macropixel_size
        if not isinstance(size, int) or size <= 0 or any(n % size for n in self.dmd_shape):
            raise ValueError("macropixel_size must be positive and divide both dimensions.")
        return self.dmd_shape[0] // size, self.dmd_shape[1] // size

    def expand_macropixels(self, grid, macropixel_size):
        """Expand each control value into a square block on the DMD."""
        grid = _binary_array(grid, self.macropixel_shape(macropixel_size))
        return np.repeat(np.repeat(grid, macropixel_size, axis=0), macropixel_size, axis=1)

    def region_masks(self, focus_size, background_size):
        """Build a centred focus square and four disjoint background regions."""
        if any(not isinstance(size, int) or size <= 0 or size % 2 == 0
               for size in (focus_size, background_size)):
            raise ValueError("Region sizes must be positive odd integers.")
        if background_size <= focus_size:
            raise ValueError("The background square must be larger than the focus.")
        half = background_size // 2
        if any(p - half < 0 or p + half >= n
               for p, n in zip(self.target_position, self.camera_shape)):
            raise ValueError("Both region squares must fit inside the camera.")

        rows, columns = np.indices(self.camera_shape)
        row, column = self.target_position
        distances = np.maximum(np.abs(rows - row), np.abs(columns - column))
        focus = distances <= focus_size // 2
        background = (distances <= half) & ~focus
        # Top/left use indices below the target; axis pixels belong to bottom/right.
        top, left = rows < row, columns < column
        masks = {
            "focus": focus,
            "top_left": background & top & left,
            "top_right": background & top & ~left,
            "bottom_left": background & ~top & left,
            "bottom_right": background & ~top & ~left,
        }
        if any(not masks[name].any() for name in BACKGROUND_REGIONS):
            raise ValueError("Every background region must contain pixels.")
        return masks

    def enhancement(self, image, focus_size, background_size):
        """Return enhancement, focus mean, four background means, and their reference."""
        image = np.asarray(image)
        if image.shape != self.camera_shape or not np.isfinite(image).all() or (image < 0).any():
            raise ValueError("Expected a finite, nonnegative camera image of the correct shape.")
        return _image_statistics(image, self.region_masks(focus_size, background_size))


def _average_capture(bench, pattern, shots):
    """Average fresh camera shots before calculating enhancement."""
    image = np.zeros(bench.camera_shape)
    for _ in range(shots):
        image += bench.capture(pattern)
    return image / shots


def run_focusing_search(bench, macropixel_size, focus_size, background_size,
                        iterations, search_seed=0):
    """Flip one block per iteration using noisy comparisons, then validate afresh."""
    if not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be a positive integer.")
    grid_shape = bench.macropixel_shape(macropixel_size)
    masks = bench.region_masks(focus_size, background_size)
    search_rng = np.random.default_rng(search_seed)
    grid = search_rng.integers(0, 2, size=grid_shape, dtype=np.uint8)
    if not grid.any():
        grid.flat[search_rng.integers(grid.size)] = 1
    initial_image = _average_capture(
        bench, bench.expand_macropixels(grid, macropixel_size), SEARCH_SHOTS
    )
    initial_enhancement = _image_statistics(initial_image, masks)["enhancement"]
    history = []

    for iteration in range(1, iterations + 1):
        candidate = grid.copy()
        index = search_rng.integers(grid.size)
        candidate.flat[index] = 1 - candidate.flat[index]
        current_image = _average_capture(
            bench, bench.expand_macropixels(grid, macropixel_size), SEARCH_SHOTS
        )
        current_score = _image_statistics(current_image, masks)["enhancement"]
        candidate_score = None
        accepted = False
        if candidate.any():
            candidate_image = _average_capture(
                bench, bench.expand_macropixels(candidate, macropixel_size), SEARCH_SHOTS
            )
            candidate_score = _image_statistics(candidate_image, masks)["enhancement"]
            accepted = candidate_score > current_score
            if accepted:
                grid = candidate
        history.append({
            "iteration": iteration,
            "current_enhancement": current_score,
            "candidate_enhancement": candidate_score,
            "accepted": accepted,
        })

    final_pattern = bench.expand_macropixels(grid, macropixel_size)
    # Fresh validation avoids returning a lucky score selected during the search.
    final_image = _average_capture(bench, final_pattern, VALIDATION_SHOTS)
    final_stats = _image_statistics(final_image, masks)
    return {
        "settings": {
            "dmd_shape": bench.dmd_shape,
            "camera_shape": bench.camera_shape,
            "target_position": bench.target_position,
            "transmission_seed": bench.transmission_seed,
            "noise_seed": bench.noise_seed,
            "noise_std": bench.noise_std,
            "macropixel_size": macropixel_size,
            "focus_size": focus_size,
            "background_size": background_size,
            "iterations": iterations,
            "search_seed": search_seed,
            "shots_per_measurement": SEARCH_SHOTS,
            "validation_shots": VALIDATION_SHOTS,
        },
        "initial_enhancement": initial_enhancement,
        "final_enhancement": final_stats["enhancement"],
        "initial_image": initial_image,
        "final_image": final_image,
        "final_pattern": final_pattern,
        "history": history,
        "focus_mean": final_stats["focus_mean"],
        "background_means": final_stats["background_means"],
        "background_reference": final_stats["background_reference"],
    }
