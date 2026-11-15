"""Private R-default seeded sampling for cross-language permutation diagnostics.

Use NumPy's MT19937 engine with R's seed expansion and rejection sampling.
Only the default R >= 3.6 Mersenne-Twister/Rejection configuration is supported.
See the source references and fixed sampling examples in the fixture notes.
"""

import secrets
import warnings

import numpy as np


class _RPermutation:
    def __init__(self, seed):
        value = secrets.randbits(31) if seed is None else int(seed)
        expanded = []
        for step in range(675):
            value = (69069 * value + 1) & 0xFFFFFFFF
            if step >= 51:
                expanded.append(value)
        self._engine = np.random.MT19937()
        self._engine.state = {
            "bit_generator": "MT19937",
            "state": {"key": np.array(expanded, dtype=np.uint32), "pos": 624},
        }

    def _index(self, size):
        bits = (size - 1).bit_length()
        while True:
            draw = 0
            for _ in range(0, bits + 1, 16):
                draw = (draw << 16) | (int(self._engine.random_raw()) >> 16)
            draw &= (1 << bits) - 1
            if draw < size:
                return draw

    def permute(self, values):
        # R sample(x) interprets a single positive number >= 1 as 1:floor(x).
        scalar_range = len(values) == 1 and values[0] >= 1
        pool = list(range(1, int(values[0]) + 1)) if scalar_range else list(values)
        result = []
        for remaining in range(len(pool), 0, -1):
            index = self._index(remaining)
            result.append(pool[index])
            pool[index] = pool[remaining - 1]
        if scalar_range and len(result) > 1:
            warnings.warn(
                "R scalar sampling produces multiple replacements for one residual; "
                "only the first is used.",
                RuntimeWarning,
                stacklevel=3,
            )
            result = result[:1]
        return np.asarray(result, dtype=float)
