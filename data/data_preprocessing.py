"""
快手KuaiRec数据集预处理脚本
- 大矩阵用于训练/验证，小矩阵用于测试
- 基于训练集计算统计特征，统一编码ID
- 加入视频标签作为稀疏特征

时间因果（负采样）:
- 先按用户时间将大矩阵正样本切为训练段(前80%)/验证段(后20%)，再分别负采样。
- 训练段：流行度与「已看」集合仅来自训练段正样本，不把验证段将点击提前用于排除或统计。
- 验证段：流行度仍仅用训练段统计；排除「已看」= 训练段 ∪ 验证段正样本，避免把用户训练期已看视频采成负例。

内存：处理过程中尽早 del 原始大/小矩阵与中间表，降低峰值（仍会在末尾持有 train/val/test 三份结果直至落盘）。

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

class KuaiRecPreprocessor:
    def __init__(self, data_path='data/raw/', output_path='data/processed/'):
        self.data_path = data_path
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        self.item_tags = None

    def load_data(self):
        print("="*50)
        print("加载原始数据...")
        self.big = pd.read_csv(os.path.join(self.data_path, 'big_matrix.csv'))
        self.small = pd.read_csv(os.path.join(self.data_path, 'small_matrix.csv'))
        print(f"大矩阵: {self.big.shape}, 用户数: {self.big['user_id'].nunique()}, 视频数: {self.big['video_id'].nunique()}")
        print(f"小矩阵: {self.small.shape}, 用户数: {self.small['user_id'].nunique()}, 视频数: {self.small['video_id'].nunique()}")

    def process_item_features(self):
        """处理物品特征（视频标签），生成 tag1-tag4"""
        print("\n处理物品特征...")
        item_feat = pd.read_csv(os.path.join(self.data_path, 'item_feat.csv'))
        print("物品特征列名:", item_feat.columns.tolist())
        print("前几行:\n", item_feat.head())

        def parse_feat(feat_str):
            feat_str = str(feat_str).strip('[]')
            if feat_str == '':
                return [-1, -1, -1, -1]
            parts = [int(x.strip()) for x in feat_str.split(',') if x.strip()]
            parts = parts[:4]
            while len(parts) < 4:
                parts.append(-1)
            return parts

        tags = item_feat['feat'].apply(parse_feat).tolist()
        df_feat = pd.DataFrame(tags, columns=['tag1', 'tag2', 'tag3', 'tag4'])
        df_feat.index = item_feat['video_id']
        df_feat.index.name = 'video_id'
        df_feat = df_feat + 1
        df_feat = df_feat.astype(int)
        df_feat.to_csv(os.path.join(self.output_path, 'item_tags.csv'))
        print(f"物品标签处理完成，形状: {df_feat.shape}")
        self.item_tags = df_feat
        return df_feat

    def clip_watch_ratio(self, df):
        before = df['watch_ratio'].max()
        df['watch_ratio'] = df['watch_ratio'].clip(0, 1)
        print(f"最大比值从 {before:.4f} 修正为 {df['watch_ratio'].max():.4f}")
        return df

    def create_ctr_label(self, df):
        df['click'] = (df['watch_ratio'] > 0).astype(int)
        return df

    def create_time_features(self, df):
        df['time_dt'] = pd.to_datetime(df['time'])
        df['hour'] = df['time_dt'].dt.hour
        def hour_to_segment(h):
            if 6 <= h <= 11: return 1
            elif 12 <= h <= 17: return 2
            elif 18 <= h <= 23: return 3
            else: return 0
        df['time_segment'] = df['hour'].apply(hour_to_segment)
        return df

    def negative_sampling(
        self,
        df_pos,
        video_duration_map,
        neg_ratio=2,
        seed=42,
        popularity_counts=None,
        extra_seen_by_user=None,
    ):
        """
        基于流行度的负采样，1:neg_ratio。

        popularity_counts: 可选 Series(index=video_id)，用于候选视频分布；为 None 时用 df_pos 统计。
        extra_seen_by_user: 可选 dict[user_id, set[video_id]]，与当段正样本合并为「已看」，用于排除负例。
        """
        print(f"\n开始负采样（基于流行度，1:{neg_ratio}）...")
        np.random.seed(seed)
        if popularity_counts is None or len(popularity_counts) == 0:
            video_pop = df_pos['video_id'].value_counts()
        else:
            video_pop = popularity_counts
        pop_sum = float(video_pop.sum())
        if pop_sum <= 0:
            return df_pos.copy()
        pop_probs = video_pop / pop_sum
        all_videos = video_pop.index.tolist()
        all_probs = pop_probs.values

        user_groups = df_pos.groupby('user_id')
        neg_samples = []

        for user_id, group in user_groups:
            pos_videos = set(group['video_id'])
            if extra_seen_by_user is not None:
                pos_videos |= extra_seen_by_user.get(user_id, set())
            n_pos = len(group)
            n_neg = n_pos * neg_ratio
            candidate_indices = [i for i, vid in enumerate(all_videos) if vid not in pos_videos]
            if not candidate_indices:
                continue
            candidate_videos = [all_videos[i] for i in candidate_indices]
            candidate_probs = [all_probs[i] for i in candidate_indices]
            candidate_probs = np.array(candidate_probs) / np.sum(candidate_probs)
            if len(candidate_videos) >= n_neg:
                neg_videos = np.random.choice(candidate_videos, size=n_neg, replace=False, p=candidate_probs)
            else:
                neg_videos = np.random.choice(candidate_videos, size=n_neg, replace=True, p=candidate_probs)

            pos_times = group['time'].tolist()
            pos_timestamps = group['timestamp'].tolist()
            pos_dates = group['date'].tolist()
            for vid in neg_videos:
                idx = np.random.randint(len(pos_times))
                neg_samples.append({
                    'user_id': user_id,
                    'video_id': vid,
                    'play_duration': 0,
                    'video_duration': video_duration_map.get(vid, 0),
                    'time': pos_times[idx],
                    'date': pos_dates[idx],
                    'timestamp': pos_timestamps[idx],
                    'watch_ratio': 0.0
                })

        df_neg = pd.DataFrame(neg_samples)
        print(f"生成负样本数: {len(df_neg)}")
        df_all = pd.concat([df_pos, df_neg], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
        print(f"总样本数: {len(df_all)}, 正样本比例: {len(df_pos)/len(df_all):.4f}")
        return df_all

    def split_data(self, df):
        """按用户和时间划分训练/验证集：每个用户前 80% 训练，后 20% 验证"""
        print("\n划分数据集...")
        train_list, val_list = [], []
        for user_id, user_data in df.groupby('user_id'):
            user_data = user_data.sort_values('timestamp').reset_index(drop=True)
            n = len(user_data)
            if n == 1:
                train_list.append(user_data)
                continue
            train_end = int(n * 0.8)
            train_end = max(1, min(train_end, n - 1))
            train_list.append(user_data.iloc[:train_end])
            val_list.append(user_data.iloc[train_end:])
        train_df = pd.concat(train_list, ignore_index=True)
        val_df = (
            pd.concat(val_list, ignore_index=True)
            if val_list
            else pd.DataFrame(columns=df.columns)
        )
        print(f"训练集: {train_df.shape}, 正样本率: {train_df['click'].mean():.4f}")
        print(f"验证集: {val_df.shape}, 正样本率: {val_df['click'].mean():.4f}")
        return train_df, val_df

    def run(self):
        print("="*50)
        print("开始KuaiRec数据集预处理")
        print("="*50)

        # 1. 加载数据
        self.load_data()

        # 2. 处理物品特征
        self.process_item_features()

        # 3. 大矩阵：clip → 标签与时间特征 → 按时间切分 → 再分别负采样（避免用未来行为定义负例）
        big_pos = self.clip_watch_ratio(self.big.copy())
        del self.big
        big_pos = self.create_ctr_label(big_pos)
        big_pos = self.create_time_features(big_pos)
        train_pos, val_pos = self.split_data(big_pos)
        del big_pos

        _dur_parts = [train_pos[['video_id', 'video_duration']]]
        if len(val_pos) > 0:
            _dur_parts.append(val_pos[['video_id', 'video_duration']])
        _dur = pd.concat(_dur_parts, ignore_index=True)
        video_duration_map = _dur.groupby('video_id')['video_duration'].mean().to_dict()
        del _dur, _dur_parts

        pop_train = train_pos['video_id'].value_counts()
        train_seen_by_user = train_pos.groupby('user_id')['video_id'].apply(set).to_dict()

        print("\n[因果] 训练段负采样：流行度与已看集合仅含训练段正样本。")
        train_big = self.negative_sampling(
            train_pos,
            video_duration_map,
            neg_ratio=2,
            seed=42,
            popularity_counts=pop_train,
            extra_seen_by_user=None,
        )
        train_big = self.create_time_features(train_big)
        del train_pos

        if len(val_pos) == 0:
            val_big = pd.DataFrame(columns=train_big.columns)
        else:
            print(
                "\n[因果] 验证段负采样：流行度仅用训练段；已看 = 训练段 ∪ 验证段正样本。"
            )
            val_big = self.negative_sampling(
                val_pos,
                video_duration_map,
                neg_ratio=2,
                seed=43,
                popularity_counts=pop_train,
                extra_seen_by_user=train_seen_by_user,
            )
            val_big = self.create_time_features(val_big)
        del val_pos, train_seen_by_user, pop_train, video_duration_map

        # 5. 处理小矩阵
        small_df = self.clip_watch_ratio(self.small.copy())
        del self.small
        small_df = self.create_ctr_label(small_df)
        small_df = self.create_time_features(small_df)

        # 6. 小矩阵仅作测试集；训练集仅来自大矩阵
        print(f"小矩阵作为测试集，样本数: {len(small_df)}")

        # 7. 训练/验证/测试划分
        train_df = train_big
        val_df = val_big
        test_df = small_df

        # 8. 基于训练集计算统计特征（用户活跃度、视频热度）
        train_user_counts = train_df['user_id'].value_counts()
        train_video_counts = train_df['video_id'].value_counts()

        for df in [train_df, val_df, test_df]:
            df['user_activity_cnt'] = df['user_id'].map(train_user_counts).fillna(0)
            df['video_popularity'] = df['video_id'].map(train_video_counts).fillna(0)

        # 计算分位数边界（基于训练集）
        def get_bins(series):
            nonzero = series[series > 0]
            if len(nonzero) >= 3:
                _, bins = pd.qcut(nonzero, q=3, labels=[0,1,2], retbins=True)
                return [-np.inf] + list(bins[1:-1]) + [np.inf]
            else:
                return [-np.inf, np.inf]

        act_bins = get_bins(train_df['user_activity_cnt'])
        pop_bins = get_bins(train_df['video_popularity'])

        def map_segment(x, bins):
            if x <= 0:
                return 0
            for i in range(len(bins)-1):
                if bins[i] < x <= bins[i+1]:
                    return i
            return 0

        for df in [train_df, val_df, test_df]:
            df['user_activity_segment'] = df['user_activity_cnt'].apply(lambda x: map_segment(x, act_bins)).astype(int)
            df['video_pop_segment'] = df['video_popularity'].apply(lambda x: map_segment(x, pop_bins)).astype(int)

        # 9. 统一编码所有ID
        all_users = pd.concat([train_df['user_id'], val_df['user_id'], test_df['user_id']]).unique()
        all_videos = pd.concat([train_df['video_id'], val_df['video_id'], test_df['video_id']]).unique()
        user_encoder = LabelEncoder()
        user_encoder.fit(all_users)
        video_encoder = LabelEncoder()
        video_encoder.fit(all_videos)

        for df in [train_df, val_df, test_df]:
            df['user_id_idx'] = user_encoder.transform(df['user_id'])
            df['video_id_idx'] = video_encoder.transform(df['video_id'])

        encoders = {'user_id': user_encoder, 'video_id': video_encoder}
        with open(os.path.join(self.output_path, 'encoders.pkl'), 'wb') as f:
            pickle.dump(encoders, f)

        # 10. 合并视频标签特征
        train_df = train_df.merge(self.item_tags, left_on='video_id', right_index=True, how='left')
        val_df = val_df.merge(self.item_tags, left_on='video_id', right_index=True, how='left')
        test_df = test_df.merge(self.item_tags, left_on='video_id', right_index=True, how='left')

        for df in [train_df, val_df, test_df]:
            for col in ['tag1', 'tag2', 'tag3', 'tag4']:
                df[col] = df[col].fillna(0).astype(int)

        # 计算标签词汇表大小
        max_tag = max(train_df[['tag1', 'tag2', 'tag3', 'tag4']].max().max(), 1)
        tag_vocab = {f'tag{i}': max_tag + 1 for i in range(1, 5)}
        with open(os.path.join(self.output_path, 'tag_vocab.pkl'), 'wb') as f:
            pickle.dump(tag_vocab, f)

        # 11. 归一化连续特征
        cont_features = ['user_activity_cnt', 'video_popularity']
        scaler = StandardScaler()
        scaler.fit(train_df[cont_features])
        for df in [train_df, val_df, test_df]:
            df[cont_features] = scaler.transform(df[cont_features])
        with open(os.path.join(self.output_path, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)

        # 12. 保存数据集
        train_df.to_csv(os.path.join(self.output_path, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(self.output_path, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(self.output_path, 'test.csv'), index=False)

        feature_config = {
            'sparse_features': ['user_id_idx', 'video_id_idx', 'time_segment',
                                'user_activity_segment', 'video_pop_segment',
                                'tag1', 'tag2', 'tag3', 'tag4'],
            'dense_features': ['user_activity_cnt', 'video_popularity'],
            'label_features': ['click']
        }
        with open(os.path.join(self.output_path, 'feature_config.pkl'), 'wb') as f:
            pickle.dump(feature_config, f)

        print("\n" + "="*50)
        print("预处理完成！")
        print("="*50)
        return train_df, val_df, test_df

if __name__ == "__main__":
    preprocessor = KuaiRecPreprocessor(
        data_path='data/raw/',
        output_path='data/processed/',
    )
    train, val, test = preprocessor.run()