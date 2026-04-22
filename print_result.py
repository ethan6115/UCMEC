import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
import os

# ===== 1. 設定多個檔案路徑 =====
# 可以把你要畫的 reward.mat 全部放在這個 list 裡
mat_paths = [
    
    #IPPO
    #r'results/hotspotEnv/nlos_cluster_7e-3/rmappo/noncoop_rnn/run1/reward.mat',
    #r'results/hotspotEnv/nlos_cluster/rmappo/noncoop_rnn_cluster1/run1/reward.mat',
    #r'results/hotspotEnv/nlos_cluster/rmappo/noncoop_rnn/run1/reward.mat',
    #r'results/hotspotEnv/nlos_cluster/rmappo/noncoop_rnn/run2/reward.mat',
    #r'results/hotspotEnv/nlos_cluster/rmappo/noncoop_rnn/run3/reward.mat',
    # hierarchical
    #r'results/hotspotEnv/nlos_cluster/rmappo/noncoop_rnn_nofrontobs/baseline/reward.mat',
    #r'results/hotspotEnv/nlos_cluster/rmappo/hierarchical_hotspot_heuristic_pair_scorer_highlow/run1/reward.mat',
    #r'results/hotspotEnv/nlos_cluster/rmappo/hierarchical_hotspot_heuristic_pair_scorer_highlow/run2/reward.mat',

    # low 無pair scorer
    r'results/hotspotEnv/nlos_cluster/rmappo/hierarchical_hotspot_heuristic_highlow/HDRL/reward.mat',
    r'results/hotspotEnv/nlos_cluster/rmappo/hierarchical_hotspot_heuristic_highlow/run1/reward.mat',
    r'results/hotspotEnv/nlos_cluster/rmappo/hierarchical_hotspot_heuristic_highlow/run2/reward.mat',
    r'results/hotspotEnv/nlos_cluster/rmappo/hierarchical_hotspot_heuristic_highlow/run3/reward.mat',

    #pair scorer highlow
    #r'results/hotspotEnv/nlos_cluster/rmappo/hierarchical_heuristic_pair_scorer_highlow/run1/reward.mat',
    #r'results/hotspotEnv/nlos_cluster/rmappo/hierarchical_heuristic_pair_scorer_highlow/proposed_method/reward.mat',
    r'results/hotspotEnv/nlos_cluster/rmappo/hierarchical_heuristic_pair_scorer_highlow/run4/reward.mat',
    #r'results/hotspotEnv/nlos_cluster/rmappo/hierarchical_heuristic_pair_scorer_highlow/run5/reward.mat',

    #pair scorer(舊的，共享g)
    #r'results/hotspotEnv/nlos_cluster_7e-3/rmappo/hierarchical_hotspot_heuristic_pair_scorer/run1/high_reward.mat',
    #r'results/hotspotEnv/nlos_cluster/rmappo/hierarchical_hotspot_heuristic_pair_scorer/Pair Scorer/high_reward.mat',
    #r'results/hotspotEnv/nlos_cluster/rmappo/hierarchical_hotspot_heuristic_pair_scorer/run2/high_reward.mat',
    #r'results/hotspotEnv/nlos_cluster/rmappo/hierarchical_hotspot_heuristic_pair_scorer/run3/high_reward.mat',
    #r'results/hotspotEnv/nlos_cluster/rmappo/hierarchical_hotspot_heuristic_pair_scorer_highlow/run1/high_reward.mat',
    

    #r'results/hotspotEnv/nlos_cluster/rmappo/hierarchical_hotspot_heuristic/MLP/high_reward.mat',
    
    #low mappo實驗
    #hierachical
    #r'results/hotspotEnv/nlos_cluster_mappo/rmappo/hierarchical_heuristic_pair_scorer_highlow/run1/reward.mat',
    #r'results/hotspotEnv/nlos_cluster_mappo/rmappo/hierarchical_heuristic_pair_scorer_highlow/run2/reward.mat',
    #r'results/hotspotEnv/nlos_cluster_mappo/rmappo/hierarchical_heuristic_pair_scorer_highlow/run3/reward.mat',
    #MAPPO low only
    #r'results/hotspotEnv/nlos_cluster_mappo/rmappo/coop_rnn/MAPPO_sharedreward/reward.mat',
    #r'results/hotspotEnv/nlos_cluster_mappo/rmappo/per_reward_wrong/coop_rnn/MAPPO_perreward/reward.mat',

    #low ablation fix power
    #r'results/hotspotEnv/nlos_cluster_low_ablation/rmappo/noncoop_rnn_minpower/run1/reward.mat',
    #r'results/hotspotEnv/nlos_cluster_low_ablation/rmappo/noncoop_rnn_maxpower/run1/reward.mat',
    ## hierarchical ablation fix power
    #r'results/hotspotEnv/nlos_cluster_low_ablation/rmappo/hierarchical_heuristic_pair_scorer_highlow_minpower/minpower/reward.mat',
    #r'results/hotspotEnv/nlos_cluster_low_ablation/rmappo/hierarchical_heuristic_pair_scorer_highlow_maxpower/maxpower/reward.mat',
]

def load_reward(file_path):
    """讀取單一 reward.mat，回傳 1D np.array，失敗則回傳 None"""
    if not os.path.exists(file_path):
        print(f"錯誤: 找不到檔案 {file_path}")
        return None

    data = scio.loadmat(file_path)
    
    if 'reward' not in data and 'high_reward' not in data:
        print(f"檔案 {file_path} 中找不到 'reward' 變數，現有變數: {list(data.keys())}")
        return None
    
    if 'reward' in data:
        rewards = data['reward'].flatten()
    else:
        rewards = data['high_reward'].flatten()
    return rewards

def smooth_rewards(rewards, window_size=5):
    """簡單的移動平均平滑"""
    if len(rewards) >= window_size:
        return np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
    else:
        return rewards

def plot_rewards(file_paths, window_size=5, show_raw=False):
    """
    file_paths: list of reward.mat 路徑
    window_size: 移動平均的視窗大小
    show_raw: 是否也畫出原始曲線
    """
    plt.figure(figsize=(10, 6))

    any_valid = False  # 檢查有沒有至少一個檔案成功載入

    for idx, path in enumerate(file_paths):
        rewards = load_reward(path)
        if rewards is None:
            continue  # 讀取失敗就跳過

        rewards_smooth = smooth_rewards(rewards, window_size)

        # 用檔名(或 index)當 label
        label_base = os.path.basename(os.path.dirname(path))  # 例如 run1、run2
        if label_base == '':
            label_base = f'Run {idx+1}'

        if show_raw:
            plt.plot(rewards, alpha=0.3, label=f'{label_base} Raw')

        plt.plot(
            rewards_smooth,
            linewidth=2,
            label=f'{label_base} MA={window_size}'
        )

        any_valid = True

    if not any_valid:
        print("沒有任何有效的 reward 資料可以畫圖，請檢查路徑與 .mat 檔內容。")
        return

    plt.title("Training Convergence")
    plt.xlabel("Training Iterations")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # plt.savefig('training_result_multi.png')

if __name__ == "__main__":
    plot_rewards(mat_paths, window_size=10, show_raw=False)
