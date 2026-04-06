"""
检查小矩阵（测试集）中的 user_id / video_id 是否曾出现在「大矩阵训练切分」中。

与 data_preprocessing.py 对齐：大矩阵 clip → 标签与时间特征 → 按用户 timestamp 前 80% 训练 / 后 20% 验证 → 再分别负采样。
仅用 ID 嵌入的模型在训练阶段只会更新训练集中出现过的 ID；测试集中未出现的 ID 等价于
用户冷启动 / 物品冷启动（或跨子集评估）。

用法（在项目根目录）:
    python src/check_test_cold_start_ids.py
    python src/check_test_cold_start_ids.py --data-path data/raw --neg-ratio 2 --seed 42
    # 大矩阵很大时负采样耗时长，可先粗查（冷启动比例会偏乐观）:
    python src/check_test_cold_start_ids.py --no-neg-sampling
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_preprocessor_class(project_root: Path):
    path = project_root / "data" / "data_preprocessing.py"
    if not path.is_file():
        raise FileNotFoundError(f"未找到预处理脚本: {path}")
    spec = importlib.util.spec_from_file_location("data_preprocessing", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.KuaiRecPreprocessor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="审计小矩阵相对大矩阵训练切分的 ID 冷启动情况"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw",
        help="原始 CSV 目录（含 big_matrix.csv, small_matrix.csv）",
    )
    parser.add_argument(
        "--neg-ratio",
        type=int,
        default=2,
        help="负采样比例，须与预处理一致",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="负采样随机种子，须与预处理一致",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=20,
        help="每种情况最多打印的 ID 示例数量",
    )
    parser.add_argument(
        "--no-neg-sampling",
        action="store_true",
        help="跳过负采样（快很多）；训练集 ID 会偏少，冷启动比例偏乐观，仅作粗查",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    data_path = (project_root / args.data_path).resolve()
    if not data_path.is_dir():
        print(f"错误: 数据目录不存在: {data_path}", file=sys.stderr)
        sys.exit(1)

    KuaiRecPreprocessor = _load_preprocessor_class(project_root)
    # output_path 仅用于满足构造函数；不写业务文件
    tmp_out = project_root / "data" / "processed" / "_cold_start_check"
    pre = KuaiRecPreprocessor(data_path=str(data_path), output_path=str(tmp_out))

    print("加载 big / small 矩阵…", flush=True)
    if args.no_neg_sampling:
        print("  (--no-neg-sampling) 大矩阵仅正样本 + 80%% 切分，与完整预处理不完全一致。", flush=True)
    pre.load_data()
    print("  已加载，构建训练切分…", flush=True)
    big = pre.clip_watch_ratio(pre.big.copy())
    if args.no_neg_sampling:
        big_df = pre.create_ctr_label(big)
        big_df = pre.create_time_features(big_df)
        train_df, _val_df = pre.split_data(big_df)
    else:
        big = pre.create_ctr_label(big)
        big = pre.create_time_features(big)
        train_pos, val_pos = pre.split_data(big)
        _dur_parts = [train_pos[["video_id", "video_duration"]]]
        if len(val_pos) > 0:
            _dur_parts.append(val_pos[["video_id", "video_duration"]])
        _dur = pd.concat(_dur_parts, ignore_index=True)
        video_duration_map = _dur.groupby("video_id")["video_duration"].mean().to_dict()
        pop_train = train_pos["video_id"].value_counts()
        train_df = pre.negative_sampling(
            train_pos,
            video_duration_map,
            neg_ratio=args.neg_ratio,
            seed=args.seed,
            popularity_counts=pop_train,
            extra_seen_by_user=None,
        )
        train_df = pre.create_time_features(train_df)

    train_users = set(train_df["user_id"].unique())
    train_videos = set(train_df["video_id"].unique())

    small = pre.small.copy()
    small_users = set(small["user_id"].unique())
    small_videos = set(small["video_id"].unique())

    # 全大矩阵（任意切分）出现过的 ID — 用于区分「仅训练未见过」vs「整个大矩阵都没有」
    big_all_users = set(pre.big["user_id"].unique())
    big_all_videos = set(pre.big["video_id"].unique())

    users_missing_train = small_users - train_users
    videos_missing_train = small_videos - train_videos

    users_not_in_big = small_users - big_all_users
    videos_not_in_big = small_videos - big_all_videos

    n_small_rows = len(small)
    row_user_cold = ~small["user_id"].isin(train_users)
    row_video_cold = ~small["video_id"].isin(train_videos)
    row_either_cold = row_user_cold | row_video_cold
    row_both_seen = (~row_user_cold) & (~row_video_cold)

    def _sample_ids(s: set, k: int):
        return sorted(s)[:k]

    print("\n" + "=" * 60)
    print("1) 小矩阵中相对「大矩阵训练切分」的 ID 集合差异")
    print("=" * 60)
    print(f"训练集(大矩阵前80%/用户) 用户数: {len(train_users):,}, 视频数: {len(train_videos):,}")
    print(f"小矩阵 用户数: {len(small_users):,}, 视频数: {len(small_videos):,}")
    print(
        f"小矩阵独有用户（未在训练切分出现）: {len(users_missing_train):,} "
        f"({100 * len(users_missing_train) / max(len(small_users), 1):.2f}% of 小矩阵用户)"
    )
    print(
        f"小矩阵独有视频（未在训练切分出现）: {len(videos_missing_train):,} "
        f"({100 * len(videos_missing_train) / max(len(small_videos), 1):.2f}% of 小矩阵视频)"
    )

    print("\n" + "=" * 60)
    print("2) 小矩阵测试行级别（ID 嵌入视角）")
    print("=" * 60)
    print(f"小矩阵总行数: {n_small_rows:,}")
    print(
        f"用户 ID 在训练切分未出现 的行: {int(row_user_cold.sum()):,} "
        f"({100 * row_user_cold.mean():.2f}%)"
    )
    print(
        f"视频 ID 在训练切分未出现 的行: {int(row_video_cold.sum()):,} "
        f"({100 * row_video_cold.mean():.2f}%)"
    )
    print(
        f"用户或视频至少一方未在训练切分出现: {int(row_either_cold.sum()):,} "
        f"({100 * row_either_cold.mean():.2f}%)"
    )
    print(
        f"用户与视频均在训练切分出现过: {int(row_both_seen.sum()):,} "
        f"({100 * row_both_seen.mean():.2f}%)"
    )

    print("\n" + "=" * 60)
    print("3) 相对「整表 big_matrix」是否也未出现（跨子集 / 真·新 ID）")
    print("=" * 60)
    print(
        f"小矩阵用户里完全不在 big_matrix 的: {len(users_not_in_big):,} "
        f"(subset of 上面「训练未见过用户」)"
    )
    print(
        f"小矩阵视频里完全不在 big_matrix 的: {len(videos_not_in_big):,} "
        f"(subset of 上面「训练未见过视频」)"
    )

    k = max(0, args.max_examples)
    if k:
        print("\n" + "-" * 60)
        print(f"示例 ID（每类最多 {k} 个，已排序）")
        print("-" * 60)
        if users_missing_train:
            print(f"训练未出现的 user_id: {_sample_ids(users_missing_train, k)}")
        if videos_missing_train:
            print(f"训练未出现的 video_id: {_sample_ids(videos_missing_train, k)}")
        if users_not_in_big:
            print(f"big 中也没有的 user_id: {_sample_ids(users_not_in_big, k)}")
        if videos_not_in_big:
            print(f"big 中也没有的 video_id: {_sample_ids(videos_not_in_big, k)}")

    print("\n结论: 若「训练切分未出现」比例显著 > 0，则测试评估包含用户/物品冷启动；")
    print("      仅用 ID 嵌入时这些行的表示主要依赖初始化/侧信息，而非交互学到的 embedding。")


if __name__ == "__main__":
    main()
