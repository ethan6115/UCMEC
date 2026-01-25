import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
import os

# ===== 1. 設定多個檔案路徑 =====
# 可以把你要畫的 reward.mat 全部放在這個 list 裡
mat_paths = [
    #IPPO
    r'C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\newEnv\MyEnv\rmappo\noncoop_rnn\IPPO_cluster5\reward.mat',

    #hierarchical
    r'C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\newEnv\MyEnv\rmappo\hierarchical_noncoop_rnn\hierarchical_IPPO_alluser_front\reward.mat',
    #r'C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\newEnv\MyEnv\rmappo\hierarchical_noncoop_rnn\hierarchical_IPPO_alluser\high_reward.mat',
    r'C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\newEnv\MyEnv\rmappo\hierarchical_noncoop_rnn\hierarchical_IPPO_alluser\reward.mat',
    #r'C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\newEnv\MyEnv\rmappo\hierarchical_noncoop_rnn\hierarchical_IPPO_alluser_attset\high_reward.mat',
    r'C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\newEnv\MyEnv\rmappo\hierarchical_noncoop_rnn\run1\reward.mat',
    #r'C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\newEnv\MyEnv\rmappo\hierarchical_noncoop_rnn\run1\high_reward.mat',


    #hierarchical peruser
    #r'C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\newEnv\MyEnv\rmappo\hierarchical_noncoop_rnn\hierarchical_IPPO_peruser\reward.mat'
    #r'C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\newEnv\MyEnv\rmappo\hierarchical_noncoop_rnn\run1\high_reward.mat'



    #r'C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\MyEnv\MyEnv\rmappo\coop_rnn_test\MAPPO_coop_commlim\reward.mat',
    #r'C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\MyEnv\MyEnv\rmappo\coop_rnn_test\MAPPO_biggermodel\reward.mat'

    #r'C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\MyEnv\MyEnv\rmappo\noncoop_mappo_test\MAPPO_noncoop_commlim\reward.mat'
    # r'... 再加其他檔案路徑',
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
    plot_rewards(mat_paths, window_size=50, show_raw=False)
