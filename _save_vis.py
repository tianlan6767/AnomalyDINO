"""把 anomaly_maps 里的 .npy 叠到原图上，存成 JPG。"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.detection import _imread_rgb
from src.utils import dists2map

RES = Path(__file__).resolve().parent / (
    r"results_tiles768\dinov2_vits14_448"
    r"\-1-shot_preprocess=force_no_mask_no_rotation_top_phasew_768"
)
MAPS = RES / "anomaly_maps" / "seed=0"
ROOT = Path(
    r"\\192.168.110.142\shareD\Project\9601一代机AOI"
    r"\04-TM-散热片-成品AOI\03_task\ader_task_top_phasew\01_images_768"
)
OUT = RES / "vis"


def main() -> None:
    scores = {}
    with (RES / "measurements_seed=0.csv").open(encoding="gbk", newline="") as f:
        for row in csv.DictReader(f):
            scores[(row["Object"], row["Sample"])] = float(row["Anomaly_Score"])
    n = 0
    for npy in sorted(MAPS.rglob("*.npy")):
        rel = npy.relative_to(MAPS)
        cls, split, kind, name = rel.parts
        src = ROOT / cls / split / kind / f"{Path(name).stem}.jpg"
        if not src.is_file():
            print("missing", src, flush=True)
            continue
        img = _imread_rgb(str(src))
        amap = dists2map(np.load(npy).astype(np.float32), img.shape)
        heat = amap / (float(amap.max()) + 1e-8)
        heat = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        blend = (0.55 * img + 0.45 * heat).astype(np.uint8)
        score = scores.get((cls, f"{kind}/{Path(name).stem}.jpg"), None)
        text = f"{cls} {kind} score={score:.3f}" if score is not None else f"{cls} {kind}"
        cv2.putText(blend, text, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        dst = OUT / cls / kind / f"{Path(name).stem}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        bgr = cv2.cvtColor(blend, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise RuntimeError(dst)
        buf.tofile(str(dst))
        n += 1
        if n % 100 == 0:
            print(n, flush=True)
    print("saved", n, OUT, flush=True)


if __name__ == "__main__":
    main()
