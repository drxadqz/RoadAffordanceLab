# 数据与复现指南

[English](data_and_reproduction.md) | **简体中文**

## 1. 公开与非公开内容

仓库公开训练、验证和测试代码、配置、选定 checkpoint、机器可读指标、历史 manifest 压缩包、环境记录与 checkpoint 谱系。RSCD 原始图片不在本仓库中重新分发，使用者需要按照数据集原始许可自行获取。

## 2. 环境安装

```bash
git clone https://github.com/drxadqz/RoadAffordanceLab.git
cd RoadAffordanceLab
git lfs install
git lfs pull

conda env create -f environment-faf-paper.yml
conda activate faf_paper
pip install -e .
```

也可以使用：

```bash
pip install -r requirements.txt
pip install -e .
```

## 3. 数据路径配置

复制公开模板：

```bash
cp configs/data/local_paths.example.yaml configs/data/local_paths.yaml
```

编辑 `configs/data/local_paths.yaml`，使其指向本机 RSCD 数据。不要把包含本机绝对路径、用户名或私有数据位置的文件提交到 GitHub。

## 4. Manifest 格式

训练代码使用 CSV manifest。核心字段为：

```text
image_path,split,dataset,class_label,domain_id,
friction_label,material_label,unevenness_label,
wetness_label,snow_label,risk_label,mu_low,mu_high
```

- `class_label`：27 类组合标签；
- `friction_label`：dry、wet、water、snow、ice 等状态因子；
- `material_label`：asphalt、concrete、mud、gravel 或 none；
- `unevenness_label`：smooth、slight、severe 或 none；
- 其余字段为领域、风险区间和辅助监督信息。

构建 manifest：

```bash
python scripts/build_manifests.py \
  --config configs/data/local_paths.yaml \
  --out-dir data/manifests_full
```

## 5. 训练

```bash
python train.py --config configs/c3_farnet/current_best_s7_public.yaml
```

公开 S7 配置会引用 Git-LFS 下的 warm-start checkpoint 与 teacher。它不是一个“只下载随机初始化模型并训练一轮即可复现”的实验。正式复现前请检查：

1. `git lfs pull` 已完成；
2. 三份 manifest 与配置引用的类别顺序一致；
3. checkpoint 文件 SHA 和[发布清单](s7_release_inventory.md)一致；
4. 环境与 `environment-faf-paper.yml` 或 `recovery/environment` 中记录相符；
5. 只使用 validation 进行 checkpoint 选择，test 不参与结构和超参数决策。

## 6. 验证和测试

```bash
python validate.py \
  --config configs/c3_farnet/current_best_s7_public.yaml \
  --checkpoint checkpoints/c3_farnet_formal_fullmanifest_source_reliable_router_s7_20260709/best_checkpoint.pth \
  --output-dir outputs/reproduce_s7_val

python test.py \
  --config configs/c3_farnet/current_best_s7_public.yaml \
  --checkpoint checkpoints/c3_farnet_formal_fullmanifest_source_reliable_router_s7_20260709/best_checkpoint.pth \
  --output-dir outputs/reproduce_s7_test
```

## 7. 报告规范

至少同时报告：

- Top-1；
- Macro-F1；
- 每个类别的 precision、recall 和 F1；
- 最弱类别 F1；
- wet/water 与 slight/severe 困难类别对；
- 单模型还是集成、是否使用 TTA、输入尺寸和 checkpoint 选择规则。

小型代表性子集仅用于小时级机制筛选，不能直接替代 49,500 张历史正式测试协议，也不能把子集结果写成全量 RSCD SOTA。

## 8. 恢复材料的边界

`recovery/` 保存了另一台训练电脑能够恢复的历史源码、manifest、环境和命令证据。部分更早的 exact anchor 目录为空，因此仓库明确区分：

- 已经存在且可校验的二进制文件；
- 根据同族 checkpoint 恢复的状态；
- 只有目录或日志证据、但原始权重已经缺失的环节。

详细说明见 [`recovery/RECOVERY_STATUS.md`](../recovery/RECOVERY_STATUS.md)。
