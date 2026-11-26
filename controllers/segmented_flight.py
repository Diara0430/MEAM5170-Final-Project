"""Fit an incoming ballistic track with a shared velocity across camera position shifts."""
import numpy as np


class SegmentedFlightFit:
    """Use acquisition timestamps. Each detected shift introduces a free position offset."""
    def __init__(self):
        self.breaks = []
        self.change_count = 0

    @staticmethod
    def _line(t, positions):
        centered = t - t.mean()
        velocity = centered @ positions / (centered @ centered)
        return positions.mean(axis=0) - t.mean() * velocity, velocity

    def fit(self, now, dt, positions, noise_m):
        # Gravity has already been removed. A calibration shift affects the
        # intercept, while all segments continue to constrain the same velocity.
        stamps = now + dt
        self.breaks = [b for b in self.breaks if b > stamps[0] + 1e-8]
        start = np.searchsorted(stamps, self.breaks[-1] - 1e-8) if self.breaks else 0
        t, y = dt[start:], positions[start:]
        if len(t) >= 12:
            past, recent = t[:-3], t[-3:]
            intercept, slope = self._line(past, y[:-3])
            offset = y[-3:].mean(axis=0) - (intercept + recent.mean() * slope)
            centered = past - past.mean()
            variance_factor = 1/3 + 1/len(past) + (recent.mean()-past.mean())**2/(centered @ centered)
            sigma = max(noise_m, .0005)
            magnitude = np.linalg.norm(offset)
            # Two consistent innovations keep an isolated bad frame from moving the origin.
            last_two = y[-2:] - (intercept + np.outer(t[-2:], slope))
            aligned = np.all(last_two @ offset > 2 * sigma * magnitude)
            if aligned and magnitude > 5 * sigma * np.sqrt(variance_factor):
                # Choose the best-supported change among the last few delivered frames.
                costs = []
                for cut in range(len(t)-5, len(t)-1):
                    groups = (slice(None,cut),slice(cut,None))
                    tx = np.concatenate([t[g]-t[g].mean() for g in groups])
                    yy = np.concatenate([y[g]-y[g].mean(axis=0) for g in groups])
                    residual = yy - np.outer(tx, tx @ yy / (tx @ tx))
                    costs.append(np.sum(residual**2))
                cut = len(t)-5 + int(np.argmin(costs))
                self.breaks.append(stamps[start+cut])
                self.change_count += 1
        if not self.breaks:
            return self._line(dt, positions)
        edges = np.searchsorted(stamps, np.asarray(self.breaks)-1e-8)
        times = np.split(dt, edges)
        points = np.split(positions, edges)
        centered = np.concatenate([g-g.mean() for g in times])
        values = np.concatenate([g-g.mean(axis=0) for g in points])
        velocity = centered @ values / (centered @ centered)
        position = points[-1].mean(axis=0) - times[-1].mean() * velocity
        return position, velocity
