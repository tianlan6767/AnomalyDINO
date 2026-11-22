"""按 infer_one.py 同一套 4-shot 配置，用良品图像分数写出标定文件。

建库用 train/good 排序后的前 N 张。这 N 张在库里，k=1 分数接近 0，不参与定阈值。
其余良品用 k=1、前 1% patch 距离均值，取分位数作为 img_thr。

    python calibrate.py --cls bottle --out calibration_bottle.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.backbones import get_model
from src.detection import _imread_rgb
from src.post_eval import mean_top1p
from src.utils import augment_image, get_dataset_info, list_images

_ORIG = torch.hub.load
_LOCAL = Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main"


def _hub(repo, model, *args, **kwargs):
    if repo == "facebookresearch/dinov2":
        return _ORIG(str(_LOCAL), model, source="local", *args, **kwargs)
    return _ORIG(repo, model, *args, **kwargs)


torch.hub.load = _hub


def _score(model, index, rgb, use_mask: bool) -> float:
    ten, grid = model.prepare_image(rgb)
    feat = model.extract_features(ten).astype("float32")
    keep = model.compute_background_mask(feat, grid, threshold=10, masking_type=use_mask)
    query = feat[keep]
    faiss.normalize_L2(query)
    dist, _ = index.search(query, 1)
    output = np.zeros(feat.shape[0], dtype=np.float32)
    output[keep] = dist[:, 0] / 2.0
    return float(mean_top1p(output))


def main() -> None:
    p = argparse.ArgumentParser(description="AnomalyDINO 良品标定，写出 img_thr")
    p.add_argument("--cls", required=True)
    p.add_argument("--data-root", type=Path, default=Path(r"Z:\DataSet\ADDataSet\MVTecAD"))
    p.add_argument("--ok-dir", type=Path, default=None, help="标定 OK 目录，默认 <cls>/train/good 里不在记忆库的图")
    p.add_argument("--shots", type=int, default=4)
    p.add_argument("--percentile", type=float, default=99.5)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    good = args.data_root / args.cls / "train" / "good"
    if not good.is_dir():
        raise SystemExit(f"找不到良品目录: {good}")
    objects, _, masking, rotation = get_dataset_info("MVTec", "agnostic", data_path=str(args.data_root))
    if args.cls not in objects:
        raise SystemExit(f"未知类别 {args.cls}，可选: {', '.join(objects)}")
    use_mask = bool(masking[args.cls])
    use_rot = bool(rotation[args.cls])

    names = list_images(str(good))
    refs = names if args.shots < 0 else names[: args.shots]
    ok_dir = args.ok_dir or good
    if args.ok_dir is None:
        ok_names = names[len(refs):]
    else:
        ok_names = list_images(str(ok_dir))
    if not ok_names:
        raise SystemExit("没有可用于标定的良品。4-shot 建库的那几张不能拿来定阈值，请提供额外 OK 目录 --ok-dir。")

    model = get_model("dinov2_vits14", "cuda", smaller_edge_size=448)
    feats = []
    for name in refs:
        rgb = _imread_rgb(str(good / name))
        views = augment_image(rgb) if use_rot else [rgb]
        for view in views:
            ten, grid = model.prepare_image(view)
            feat = model.extract_features(ten)
            keep = model.compute_background_mask(feat, grid, threshold=10, masking_type=False)
            feats.append(feat[keep].astype("float32"))
    bank = np.concatenate(feats, 0)
    faiss.normalize_L2(bank)
    index = faiss.IndexFlatL2(bank.shape[1])
    index.add(bank)

    scores = []
    for i, name in enumerate(ok_names):
        scores.append(_score(model, index, _imread_rgb(str(ok_dir / name)), use_mask))
        if (i + 1) % 10 == 0 or i + 1 == len(ok_names):
            print(f"  calibrate {i + 1}/{len(ok_names)}", flush=True)
    arr = np.asarray(scores, dtype=np.float64)
    img_thr = float(np.percentile(arr, args.percentile))
    payload = {
        "cls": args.cls,
        "img_score": "meantop1p_k1",
        "img_thr": img_thr,
        "percentile": float(args.percentile),
        "ratio": 0.8,
        "shots": len(refs),
        "mask": use_mask,
        "rotate": use_rot,
        "n_ok": len(ok_names),
        "ok_sp_min": float(arr.min()),
        "ok_sp_max": float(arr.max()),
        "ok_sp_p50": float(np.percentile(arr, 50)),
        "ok_dir": str(ok_dir),
        "excluded_refs": refs,
        "model": "dinov2_vits14_448",
        "preprocess": "agnostic",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out}  n_ok={len(ok_names)} img_thr={img_thr:.6g}")


if __name__ == "__main__":
    main()
