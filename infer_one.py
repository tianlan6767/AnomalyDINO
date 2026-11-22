"""单张图测试：按官方 MVTec 这次的 4-shot 设置建库，写出 pred_mask。

    python infer_one.py --cls bottle --image Z:/.../MVTecAD/bottle/test/broken_large/000.png

输出默认在图片旁边：<原名>_pred_mask.png，以及 <原名>_heatmap.jpg。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import faiss
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.backbones import get_model
from src.detection import _imread_rgb
from src.post_eval import mean_top1p
from src.utils import augment_image, dists2map, get_dataset_info, list_images

_ORIG = torch.hub.load
_LOCAL = Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main"


def _hub(repo, model, *args, **kwargs):
    if repo == "facebookresearch/dinov2":
        return _ORIG(str(_LOCAL), model, source="local", *args, **kwargs)
    return _ORIG(repo, model, *args, **kwargs)


torch.hub.load = _hub


def _save(path: Path, bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, bgr)
    if not ok:
        raise RuntimeError(path)
    buf.tofile(str(path))


def main() -> None:
    p = argparse.ArgumentParser(description="输入一张图，输出 pred_mask")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--cls", required=True, help="MVTec 类别，如 bottle")
    p.add_argument("--data-root", type=Path, default=Path(r"Z:\DataSet\ADDataSet\MVTecAD"))
    p.add_argument("--shots", type=int, default=4)
    p.add_argument("--out", type=Path, default=None, help="pred_mask 路径，默认 <image>_pred_mask.png")
    args = p.parse_args()
    image = args.image
    if not image.is_file():
        raise SystemExit(f"找不到图片: {image}")
    good = args.data_root / args.cls / "train" / "good"
    if not good.is_dir():
        raise SystemExit(f"找不到良品目录: {good}")

    objects, _, masking, rotation = get_dataset_info("MVTec", "agnostic", data_path=str(args.data_root))
    if args.cls not in objects:
        raise SystemExit(f"未知类别 {args.cls}，可选: {', '.join(objects)}")
    use_mask = bool(masking[args.cls])
    use_rot = bool(rotation[args.cls])

    refs = list_images(str(good))
    if args.shots > 0:
        refs = refs[: args.shots]
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

    rgb = _imread_rgb(str(image))
    ten, grid = model.prepare_image(rgb)
    feat = model.extract_features(ten).astype("float32")
    keep = model.compute_background_mask(feat, grid, threshold=10, masking_type=use_mask)
    query = feat[keep]
    faiss.normalize_L2(query)
    dist, _ = index.search(query, 1)
    dist = dist[:, 0] / 2.0
    output = np.zeros(feat.shape[0], dtype=np.float32)
    output[keep] = dist
    d_masked = output.reshape(grid)
    amap = dists2map(d_masked, rgb.shape)
    heat = amap / (float(np.max(amap)) + 1e-8)
    pred = np.where(heat > 0.8, 255, 0).astype(np.uint8)
    score = float(mean_top1p(output))

    out = args.out or image.with_name(image.stem + "_pred_mask.png")
    _save(out, pred)
    color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    blend = cv2.addWeighted(bgr, 0.55, color, 0.45, 0)
    heat_path = out.with_name(out.stem.replace("_pred_mask", "") + "_heatmap.jpg")
    if heat_path == out:
        heat_path = out.with_name(image.stem + "_heatmap.jpg")
    _save(heat_path, blend)
    print(f"cls={args.cls} shots={len(refs)} mask={use_mask} rotate={use_rot} score={score:.4f}")
    print(f"pred_mask {out}")
    print(f"heatmap   {heat_path}")


if __name__ == "__main__":
    main()
