import os
import scipy.io as scio
import numpy as np

# If matplotlib default config dir is not writable, force a writable fallback.
# This avoids unstable backend/config behavior across runs.
_mpl_cfg = os.path.expanduser("~/.config/matplotlib")
if not os.access(_mpl_cfg, os.W_OK):
    _fallback_cfg = os.path.join("/tmp", "matplotlib")
    os.makedirs(_fallback_cfg, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", _fallback_cfg)

import matplotlib.pyplot as plt

# ===== 1. 設定多個檔案路徑 =====
# 可以把你要畫的 reward.mat 全部放在這個 list 裡
mat_paths = [

    #新的
    #new reward
    #r'results/MyEnv/nlos_cluster_v2/rmappo/noncoop_rnn/run2/reward.mat',
    #best front
    #r'results/MyEnv/nlos_cluster_v2/rmappo/noncoop_rnn_bestfront/run1/reward.mat',
    #cluster1
    #r'results/MyEnv/nlos_cluster_v2/rmappo/noncoop_rnn_cluster1/run2/reward.mat',
    #cluster3
    #r'results/MyEnv/nlos_cluster_v2/rmappo/noncoop_rnn_cluster3/run1/reward.mat',
    #flat drl
    #r'results/MyEnv/nlos_cluster_v2/rmappo/flat_drl/run3/reward.mat',
    # w/o front obs
    #r'results/MyEnv/nlos_cluster/rmappo/noncoop_rnn_nofrontobs/run1/reward.mat',
    #r'results/MyEnv/nlos_cluster/rmappo/noncoop_rnn_nofrontobs/run2/reward.mat',
    #r'results/MyEnv/nlos_cluster/rmappo/noncoop_rnn_nofrontobs/run3/reward.mat',

    #hierarchical highlow
    #r'results/MyEnv/nlos_cluster/rmappo/hierarchical_pair_scorer_highlow/run1/reward.mat',
    #r'results/MyEnv/nlos_cluster/rmappo/hierarchical_pair_scorer_highlow/run2/reward.mat',
    #new reward
    #r'results/MyEnv/nlos_cluster_v2/rmappo/hierarchical_pair_scorer_highlow/run3/reward.mat',
    #r'results/MyEnv/nlos_cluster_v2/rmappo/hierarchical_pair_scorer_highlow/run3/reward.mat',
    #r'results/MyEnv/nlos_cluster_v2/rmappo/hierarchical_pair_scorer_highlow/run4/reward.mat',
    #r'results/MyEnv/nlos_cluster_v2/rmappo/hierarchical_pair_scorer_highlow_meanmaxglobal/run1/reward.mat',


    #w/o pair scorer
    #r'results/MyEnv/nlos_cluster_high_ablation_v2/rmappo/hierarchical_highlow/run3/reward.mat',

    #fix power
    #r'results/MyEnv/nlos_cluster_high_ablation_v2/rmappo/hierarchical_pair_scorer_highlow_minpower/run5/reward.mat',
    #r'results/MyEnv/nlos_cluster_high_ablation_v2/rmappo/hierarchical_pair_scorer_highlow_maxpower/run5/reward.mat',
    #r'results/MyEnv/nlos_cluster_low_ablation/rmappo/hierarchical_pair_scorer_highlow_minpower/run1/reward.mat',
    #r'results/MyEnv/nlos_cluster_low_ablation/rmappo/hierarchical_pair_scorer_highlow_maxpower/run1/reward.mat',

    #high ablation
    #pair concat
    #r'results/MyEnv/nlos_cluster_high_ablation_v2/rmappo/hierarchical_pair_scorer_pairconcat/run4/reward.mat',
    #no global
    r'results/MyEnv/nlos_cluster_high_ablation_v2/rmappo/hierarchical_pair_scorer_noglobal/run2/reward.mat',
    #candidate 5
    r'results/MyEnv/nlos_cluster_high_ablation_v2/rmappo/hierarchical_pair_scorer_noglobal_candidate5/run1/reward.mat',
    r'results/MyEnv/nlos_cluster_high_ablation_v2/rmappo/hierarchical_pair_scorer_noglobal_candidate8/run1/reward.mat',
    r'results/MyEnv/nlos_cluster_high_ablation_v2/rmappo/hierarchical_pair_scorer_noglobal_candidate12/run1/reward.mat',
    r'results/MyEnv/nlos_cluster_high_ablation_v2/rmappo/hierarchical_pair_scorer_noglobal_candidate15/run1/reward.mat',
    


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


def _stabilize_figure_window(fig, margin=40):
    """Center the figure window and clamp size to visible screen area."""
    try:
        manager = fig.canvas.manager
        window = manager.window
    except Exception:
        return  # Non-GUI backend (e.g., Agg) or no window handle.

    try:
        # Tk backend
        if hasattr(window, "winfo_screenwidth") and hasattr(window, "geometry"):
            window.update_idletasks()
            sw = int(window.winfo_screenwidth())
            sh = int(window.winfo_screenheight())
            ww = int(window.winfo_width())
            wh = int(window.winfo_height())
            if ww <= 1 or wh <= 1:
                fw, fh = fig.get_size_inches()
                dpi = fig.get_dpi()
                ww = int(fw * dpi)
                wh = int(fh * dpi)
            ww = max(300, min(ww, max(300, sw - 2 * margin)))
            wh = max(220, min(wh, max(220, sh - 2 * margin)))
            x = max(0, (sw - ww) // 2)
            y = max(0, (sh - wh) // 2)
            window.geometry(f"{ww}x{wh}+{x}+{y}")
            return

        # Qt backend
        if hasattr(window, "screen") and hasattr(window, "move") and hasattr(window, "resize"):
            screen = window.screen()
            if screen is not None:
                geo = screen.availableGeometry()
                sw = int(geo.width())
                sh = int(geo.height())
                sx = int(geo.x())
                sy = int(geo.y())
            else:
                sw, sh, sx, sy = 1920, 1080, 0, 0
            ww = int(window.width())
            wh = int(window.height())
            if ww <= 1 or wh <= 1:
                fw, fh = fig.get_size_inches()
                dpi = fig.get_dpi()
                ww = int(fw * dpi)
                wh = int(fh * dpi)
            ww = max(300, min(ww, max(300, sw - 2 * margin)))
            wh = max(220, min(wh, max(220, sh - 2 * margin)))
            x = sx + max(0, (sw - ww) // 2)
            y = sy + max(0, (sh - wh) // 2)
            window.resize(ww, wh)
            window.move(x, y)
            return

        # Fallback for other backends
        if hasattr(window, "move"):
            window.move(80, 60)
    except Exception:
        # If the backend does not support positioning, keep default behavior.
        pass

def plot_rewards(file_paths, window_size=5, show_raw=False):
    """
    file_paths: list of reward.mat 路徑
    window_size: 移動平均的視窗大小
    show_raw: 是否也畫出原始曲線
    """
    fig, ax = plt.subplots(num="Training Convergence", figsize=(10, 6), clear=True)
    _window_stabilized = {"done": False}

    def _on_first_draw(_event):
        if _window_stabilized["done"]:
            return
        _window_stabilized["done"] = True
        _stabilize_figure_window(fig)

    fig.canvas.mpl_connect("draw_event", _on_first_draw)

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
            ax.plot(rewards, alpha=0.3, label=f'{label_base} Raw')

        ax.plot(
            rewards_smooth,
            linewidth=2,
            label=f'{label_base} MA={window_size}'
        )

        any_valid = True

    if not any_valid:
        print("沒有任何有效的 reward 資料可以畫圖，請檢查路徑與 .mat 檔內容。")
        return

    ax.set_title("Training Convergence")
    ax.set_xlabel("Training Iterations")
    ax.set_ylabel("Average Reward")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    plt.show()
    # plt.savefig('training_result_multi.png')

if __name__ == "__main__":
    plot_rewards(mat_paths, window_size=10, show_raw=False)
