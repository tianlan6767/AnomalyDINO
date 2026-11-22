# AnomalyDINO 图像分数与 pred_mask

官方 MVTec 这次的设置：`dinov2_vits14`，输入 448，4-shot，`knn_metric=L2_normalized`，`k=1`。结果目录：

`results_MVTec/dinov2_vits14_448/4-shot_preprocess=agnostic_mvtec_official_4shot`

图像级指标在同级的 `metrics_seed=0.json`。热力图和 `pred_mask` 在 `anomaly_maps/seed=0`。

一张图分两路。图像分数用来判 OK/NG，热力图和 `pred_mask` 只标出这张图里相对最热的位置。

## 需要的东西

- 待测图
- 类别名，以及该类良品目录 `<data_root>/<类别>/train/good`
- 本地 DINOv2 仓库和 `dinov2_vits14_pretrain.pth`
- faiss（这次用的是 `faiss-cpu`）

没有单独保存的训练权重。每次推理都用良品重新建库。4-shot 取 `train/good` 按文件名排序后的前 4 张。`agnostic` 下每张良品再做 8 个角度。`capsule`、`hazelnut`、`pill`、`screw`、`toothbrush` 建库时丢掉背景 patch，其余类别用整图 patch。

## 特征

图缩到 448×448。patch 边长 14，得到 32×32 = 1024 个 patch，每个 384 维。良品 patch 做 L2 归一化后放进 faiss。

## 每个 patch 的距离

测试图同样得到 1024 个向量并归一化。每个 patch 在库里找最近的 1 个良品 patch。faiss 返回平方 L2，代码再除以 2，等于 `1 - 余弦相似度`，大约在 0 到 2 之间。背景被丢掉的类别里，被 mask 掉的 patch 距离记 0。

对应 `src/detection.py`：`L2_normalized` 分支里 `distances = distances / 2`。

## 图像分数

把 1024 个距离从大到小排序，取前 1% 求平均。`1024 × 0.01 = 10.24`，取整是 10，即最大的 10 个距离的平均值。不足 1 个时退化为最大值。

```python
int(len(distances) * 0.01) 个最大距离的均值
```

实现是 `src/post_eval.py` 的 `mean_top1p`。这个数不除以该图最大值。OK 图整体偏小，NG 图整体偏大。`metrics_seed=0.json` 的图像 AUROC 用的是这个连续分数。

是否判为缺陷，要和训练集 `good` 的分位数比，超过才算 NG。测试集不参与定阈值。

## 热力图和 pred_mask

32×32 距离图双线性放大到原图尺寸，再做 σ=4 的高斯平滑，得到异常图。实现是 `src/utils.py` 的 `dists2map`。

`pred_mask`：异常图除以自己的最大值，大于 0.8 的像素写成 255，其余为 0。热力图是同一张归一化图的 JET 伪彩色，按 0.55 原图 + 0.45 彩色叠加。

除以自己的最大值之后，最大值变成 1，所以 OK 图也会有像素大于 0.8。`pred_mask` 不能用来判 OK/NG。
