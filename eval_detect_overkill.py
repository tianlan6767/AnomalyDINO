"""按点位用训练集 good 的分位数评估 AnomalyDINO 检出率和过杀率。

训练集分数取第二近邻，避免良品图命中自己后分数接近 0。
测试分数直接读已有 measurements CSV。
"""
from __future__ import annotations

import argparse
import csv
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
from src.utils import list_images

_ORIG = torch.hub.load
_LOCAL = Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main"


def _hub(repo, model, *args, **kwargs):
    if repo == "facebookresearch/dinov2":
        return _ORIG(str(_LOCAL), model, source="local", *args, **kwargs)
    return _ORIG(repo, model, *args, **kwargs)


torch.hub.load = _hub

ROOT = Path(
    r"\\192.168.110.142\shareD\Project\9601一代机AOI"
    r"\04-TM-散热片-成品AOI\03_task\ader_task_top_phasew\01_images_768"
)
MEAS = Path(__file__).resolve().parent / (
    r"results_tiles768\dinov2_vits14_448"
    r"\-1-shot_preprocess=force_no_mask_no_rotation_top_phasew_768"
    r"\measurements_seed=0.csv"
)


def _class_from_image(image: Path, cls_name: str) -> str:
    if cls_name:
        return cls_name
    for part in image.parts:
        if part.startswith("x") and "_y" in part:
            return part
    raise SystemExit("路径里没有 x*_y* 点位名，请加 --cls")


def _image_score(model, index, image: Path, k: int) -> float:
    img = _imread_rgb(str(image))
    ten, _ = model.prepare_image(img)
    feat = model.extract_features(ten).astype("float32")
    faiss.normalize_L2(feat)
    dist, _ = index.search(feat, k)
    picked = dist[:, k - 1] / 2.0
    return float(mean_top1p(picked))


def debug_one(image: Path, cls_name: str) -> None:
    """只建一个点位的良品库，打印这张图的分数和 p95 / p99.5 判定。"""
    cls = _class_from_image(image, cls_name)
    good = ROOT / cls / "train" / "good"
    if not good.is_dir():
        raise SystemExit(f"找不到良品库: {good}")
    if not image.is_file():
        raise SystemExit(f"找不到图片: {image}")
    model = get_model("dinov2_vits14", "cuda", smaller_edge_size=448)
    train = list_images(str(good))
    feats = []
    for fn in train:
        img = _imread_rgb(str(good / fn))
        ten, _ = model.prepare_image(img)
        feat = model.extract_features(ten).astype("float32")
        faiss.normalize_L2(feat)
        feats.append(feat)
    index = faiss.IndexFlatL2(feats[0].shape[1])
    index.add(np.concatenate(feats, 0))
    train_scores = []
    for feat in feats:
        dist, _ = index.search(feat, 2)
        train_scores.append(float(mean_top1p(dist[:, 1] / 2.0)))
    train_scores = np.asarray(train_scores, dtype=np.float64)
    score = _image_score(model, index, image, k=1)
    print(f"cls={cls} train={len(train_scores)} image={image}")
    print(f"score_k1={score:.6f}")
    if image.parent.resolve() == good.resolve():
        print(f"score_k2={_image_score(model, index, image, k=2):.6f}  (训练集图，k1 会命中自己)")
    for q in (95.0, 99.5):
        thr = float(np.percentile(train_scores, q))
        label = "NG" if score > thr else "OK"
        print(f"p{q:g} thr={thr:.6f} -> {label}")


def main() -> None:
    rows = list(csv.DictReader(MEAS.open(encoding="gbk")))
    by: dict[str, list[dict]] = {}
    for row in rows:
        by.setdefault(row["Object"], []).append(row)
    model = get_model("dinov2_vits14", "cuda", smaller_edge_size=448)
    qs = (95.0, 99.5)
    tot = {q: {"det": 0, "nd": 0, "ok": 0, "no": 0} for q in qs}
    per = []
    names = sorted(by)
    for i, name in enumerate(names, 1):
        train = list_images(str(ROOT / name / "train" / "good"))
        feats = []
        for fn in train:
            img = _imread_rgb(str(ROOT / name / "train" / "good" / fn))
            ten, _ = model.prepare_image(img)
            feat = model.extract_features(ten).astype("float32")
            faiss.normalize_L2(feat)
            feats.append(feat)
        index = faiss.IndexFlatL2(feats[0].shape[1])
        index.add(np.concatenate(feats, 0))
        train_scores = []
        for feat in feats:
            dist, _ = index.search(feat, 2)
            train_scores.append(float(mean_top1p(dist[:, 1] / 2.0)))
        train_scores = np.asarray(train_scores, dtype=np.float64)
        test_ok, test_ng = [], []
        for row in by[name]:
            score = float(row["Anomaly_Score"])
            if row["Sample"].startswith("good/"):
                test_ok.append(score)
            else:
                test_ng.append(score)
        test_ok = np.asarray(test_ok, dtype=np.float64)
        test_ng = np.asarray(test_ng, dtype=np.float64)
        item = {
            "cls": name,
            "n_train": int(len(train_scores)),
            "n_test_ok": int(len(test_ok)),
            "n_test_ng": int(len(test_ng)),
            "rates": [],
        }
        for q in qs:
            thr = float(np.percentile(train_scores, q))
            det = int((test_ng > thr).sum()) if len(test_ng) else 0
            over = int((test_ok > thr).sum()) if len(test_ok) else 0
            item["rates"].append({"percentile": q, "img_thr": thr, "detect": det, "overkill": over})
            tot[q]["det"] += det
            tot[q]["nd"] += len(test_ng)
            tot[q]["ok"] += over
            tot[q]["no"] += len(test_ok)
        per.append(item)
        print(
            f"[{i}/{len(names)}] {name} ng={len(test_ng)} "
            f"p95 det={item['rates'][0]['detect']} over={item['rates'][0]['overkill']} "
            f"p99.5 det={item['rates'][1]['detect']} over={item['rates'][1]['overkill']}",
            flush=True,
        )
    summary = {}
    for q in qs:
        t = tot[q]
        summary[str(q)] = {
            "detect": f"{t['det']}/{t['nd']}",
            "detect_rate": (t["det"] / t["nd"]) if t["nd"] else None,
            "overkill": f"{t['ok']}/{t['no']}",
            "overkill_rate": (t["ok"] / t["no"]) if t["no"] else None,
        }
        print("TOTAL", q, summary[str(q)], flush=True)
    outp = MEAS.parent / "detect_overkill.json"
    outp.write_text(json.dumps({"summary": summary, "per_class": per}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", outp, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AnomalyDINO 检出/过杀，或单图 debug")
    parser.add_argument("--image", type=Path, default=r"\\192.168.110.142\shareD\Project\9601一代机AOI\04-TM-散热片-成品AOI\03_task\ader_task_top_phasew\01_images_768\x06633_y04232\test\good\D4VHVP618KE0001PQH+8A26A23XH30AB000000_20260729014358_1_1_6_PMD_phasew.jpg", help="单张图。点位从路径里的 x*_y* 读取")
    parser.add_argument("--cls", default="", help="点位名，路径里没有 x*_y* 时必填")
    args = parser.parse_args()
    if args.image is not None:
        debug_one(args.image, args.cls.strip())
    else:
        main()
