"""Real-time sewing-line monitoring with OpenCV (image processing).

For every workstation ROI we run background subtraction (MOG2) and detect work bursts:
  * active segment  = operator hands / garment moving inside the ROI
  * cycle           = one completed active segment (one bundle/piece handled)
  * cycle time      = time between starts of consecutive cycles
  * utilization     = active time / elapsed time
The slowest station is flagged as the bottleneck (line-balancing signal).
The last station's cycle count is treated as line output and fed into a particle filter
that forecasts end-of-shift output.

Run:
    python -m src.vision.realtime_monitor --demo                 # synthetic video, no camera needed
    python -m src.vision.realtime_monitor --source 0 --stations 4    # webcam
    python -m src.vision.realtime_monitor --source line.mp4 --stations 6 --select-rois
Press  q  to quit.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..config import OUTPUTS_DIR
from ..forecast import ProductionParticleFilter

W, H = 640, 360


def make_grid_rois(w, h, n, margin=10):
    cw = w // n
    return [(i * cw + margin, margin + 30, cw - 2 * margin, h - 2 * margin - 30) for i in range(n)]


class StationTracker:
    def __init__(self, name, roi, thr=0.01, gap_s=0.6, min_active_s=0.8):
        self.name, self.roi, self.thr, self.gap_s, self.min_active_s = name, roi, thr, gap_s, min_active_s
        self.bg = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=40, detectShadows=True)
        self.kernel = np.ones((3, 3), np.uint8)
        self.moving = False
        self.active = False
        self.seg_start = self.last_motion = self.prev_start = None
        self.cycles, self.active_time, self.cycle_times = 0, 0.0, []
        self.t0 = None
        self.ratio = 0.0

    def update(self, frame, t):
        if self.t0 is None:
            self.t0 = t
        x, y, w, h = self.roi
        mask = (self.bg.apply(frame[y:y + h, x:x + w]) > 200).astype(np.uint8)   # drop shadows (127)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        self.ratio = float(mask.mean())
        self.moving = self.ratio > self.thr
        if self.moving:
            self.last_motion = t
            if not self.active:
                self.active, self.seg_start = True, t
        elif self.active and t - self.last_motion >= self.gap_s:
            dur = self.last_motion - self.seg_start
            if dur >= self.min_active_s:
                self.cycles += 1
                self.active_time += dur
                if self.prev_start is not None:
                    self.cycle_times.append(self.seg_start - self.prev_start)
                self.prev_start = self.seg_start
            self.active = False

    def stats(self, t):
        elapsed = max(t - (t if self.t0 is None else self.t0), 1e-6)
        act = self.active_time + (self.last_motion - self.seg_start if self.active else 0)
        return {"station": self.name, "cycles": self.cycles,
                "avg_cycle_s": round(float(np.mean(self.cycle_times)), 2) if self.cycle_times else None,
                "utilization": round(min(act / elapsed, 1.0), 3),
                "idle_ratio": round(1 - min(act / elapsed, 1.0), 3)}


class DemoSource:
    """Synthetic video: moving 'hands' in each station with different cycle periods."""
    def __init__(self, n, fps=10, seconds=180):
        self.n, self.fps, self.total = n, fps, int(fps * seconds)
        self.i = 0
        self.rng = np.random.default_rng(1)
        self.periods = [6.0, 8.0, 10.0, 7.0, 9.0, 6.5][:n] + [8.0] * max(0, n - 6)
        self.frac = [0.7, 0.6, 0.5, 0.8, 0.6, 0.7][:n] + [0.6] * max(0, n - 6)
        self.rois = make_grid_rois(W, H, n)

    def read(self):
        if self.i >= self.total:
            return False, None, None
        t = self.i / self.fps
        self.i += 1
        frame = np.full((H, W, 3), 60, np.uint8)
        frame = np.clip(frame + self.rng.normal(0, 1.5, frame.shape), 0, 255).astype(np.uint8)
        for k, (x, y, w, h) in enumerate(self.rois):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (90, 90, 90), 1)
            if t >= 3 and ((t - 3) % self.periods[k]) < self.periods[k] * self.frac[k]:
                cx = int(x + w / 2 + 0.3 * w * np.sin(2 * np.pi * 1.5 * t + k))
                cy = int(y + h / 2 + 0.25 * h * np.cos(2 * np.pi * 1.1 * t))
                cv2.circle(frame, (cx, cy), 20, (200, 170, 140), -1)
        return True, frame, t


class VideoSource:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        self.is_file = not str(source).isdigit()
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.i, self.t0 = 0, time.time()

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return False, None, None
        frame = cv2.resize(frame, (W, H))
        t = self.i / self.fps if self.is_file else time.time() - self.t0
        self.i += 1
        return True, frame, t


def draw(frame, trackers, bottleneck, forecast, t):
    for tr in trackers:
        x, y, w, h = tr.roi
        col = (0, 200, 0) if tr.active else (0, 0, 220)
        cv2.rectangle(frame, (x, y), (x + w, y + h), col, 2)
        s = tr.stats(t)
        cv2.putText(frame, f"{tr.name} c={s['cycles']}", (x + 4, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        cv2.putText(frame, f"util {s['utilization']:.0%}", (x + 4, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    txt = f"t={t:5.1f}s  bottleneck: {bottleneck or '-'}  forecast end-of-shift: {forecast:.0f}"
    cv2.putText(frame, txt, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


def run(args):
    src = DemoSource(args.stations, seconds=args.demo_seconds) if args.demo else VideoSource(args.source)
    ok, frame, t = src.read()
    if not ok:
        raise RuntimeError("No frames received")
    rois = make_grid_rois(W, H, args.stations)
    if args.select_rois and not args.demo:
        rois = [tuple(r) for r in cv2.selectROIs("Select station ROIs (ENTER=confirm, ESC=finish)", frame)]
        cv2.destroyAllWindows()
    trackers = [StationTracker(f"ST{i + 1}", r) for i, r in enumerate(rois)]
    pf = ProductionParticleFilter(init_rate=1 / 8)
    show, last_tick, last_total, mean_fc = not args.no_display, 0.0, 0, 0.0
    stats_path = OUTPUTS_DIR / "vision_stats.json"

    while ok:
        for tr in trackers:
            tr.update(frame, t)
        if t - last_tick >= 1.0:                                  # 1-second analytics tick
            total = trackers[-1].cycles                           # line output = last station
            pf.step(total - last_total, t - last_tick)
            last_total, last_tick = total, t
            mean_fc, p10, p90 = pf.forecast(total, max(args.shift_seconds - t, 0))
            rows = [tr.stats(t) for tr in trackers]
            valid = [r for r in rows if r["avg_cycle_s"]]
            bott = max(valid, key=lambda r: r["avg_cycle_s"])["station"] if valid else None
            stats_path.write_text(json.dumps({
                "time_s": round(t, 1), "line_output": total, "rate_units_per_s": round(pf.rate(), 4),
                "forecast_end_of_shift": {"mean": round(mean_fc, 1), "p10": round(p10, 1), "p90": round(p90, 1)},
                "bottleneck": bott, "stations": rows}, indent=2))
        if show:
            rows = [tr.stats(t) for tr in trackers]
            valid = [r for r in rows if r["avg_cycle_s"]]
            bott = max(valid, key=lambda r: r["avg_cycle_s"])["station"] if valid else None
            try:
                cv2.imshow("Sewing line monitor", draw(frame.copy(), trackers, bott, mean_fc, t))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except cv2.error:
                print("[warn] No GUI available - continuing headless."); show = False
        ok, frame, t = src.read()

    final = pd.DataFrame([tr.stats(t or last_tick) for tr in trackers])
    final.to_csv(OUTPUTS_DIR / "vision_summary.csv", index=False)
    print(final.to_string(index=False))
    if final["avg_cycle_s"].notna().any():
        print("Bottleneck station:", final.loc[final["avg_cycle_s"].idxmax(), "station"])
    cv2.destroyAllWindows()
    return final


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="use synthetic video")
    p.add_argument("--source", default="0", help="webcam index or video file path")
    p.add_argument("--stations", type=int, default=4)
    p.add_argument("--select-rois", action="store_true", help="draw station ROIs manually")
    p.add_argument("--no-display", action="store_true", help="headless mode")
    p.add_argument("--shift-seconds", type=float, default=240, help="shift length used for forecast")
    p.add_argument("--demo-seconds", type=float, default=180)
    return p.parse_args()


if __name__ == "__main__":
    run(parse())
