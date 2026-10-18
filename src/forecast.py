"""Particle filter that tracks the live production rate and forecasts end-of-shift output."""
import numpy as np


class ProductionParticleFilter:
    def __init__(self, n=500, init_rate=0.1, init_std=0.05, rate_std=0.01, seed=0):
        self.rng = np.random.default_rng(seed)
        self.n, self.rate_std = n, rate_std
        self.rates = np.clip(self.rng.normal(init_rate, init_std, n), 0, None)  # units / second
        self.w = np.full(n, 1.0 / n)

    def step(self, observed_units: float, dt: float):
        self.rates = np.clip(self.rates + self.rng.normal(0, self.rate_std * np.sqrt(dt), self.n), 0, None)
        expected = self.rates * dt
        sd = np.sqrt(np.maximum(expected, 0.3))                       # Poisson-like noise
        self.w *= np.exp(-0.5 * ((observed_units - expected) / sd) ** 2) / sd + 1e-300
        self.w /= self.w.sum()
        if 1.0 / np.sum(self.w ** 2) < self.n / 2:                    # systematic resampling
            idx = np.searchsorted(np.cumsum(self.w), (self.rng.random() + np.arange(self.n)) / self.n)
            self.rates = self.rates[np.minimum(idx, self.n - 1)]
            self.w = np.full(self.n, 1.0 / self.n)

    def rate(self) -> float:
        return float(np.sum(self.w * self.rates))

    def forecast(self, current_total: float, remaining_seconds: float):
        """Returns (mean, p10, p90) of end-of-shift output."""
        end = current_total + self.rates * remaining_seconds
        order = np.argsort(end); cw = np.cumsum(self.w[order])
        q = lambda p: float(end[order][min(np.searchsorted(cw, p), self.n - 1)])
        return float(np.sum(self.w * end)), q(0.1), q(0.9)
