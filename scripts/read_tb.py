"""Read tensorboard event files and print summary."""
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

base = 'c:/DCNLab/UCMEC/UCMEC-mmWave-Fronthaul/results/hotspotEnv/nlos_cluster/rmappo/hierarchical_hotspot_heuristic/run2/logs'

files = {
    'high_reward': f'{base}/average_episode_rewards_high/average_episode_rewards_high',
    'low_reward': f'{base}/average_episode_rewards/average_episode_rewards',
    'dist_entropy': f'{base}/high/dist_entropy/high/dist_entropy',
    'value_loss': f'{base}/high/value_loss/high/value_loss',
    'policy_loss': f'{base}/high/policy_loss/high/policy_loss',
    'actor_grad_norm': f'{base}/high/actor_grad_norm/high/actor_grad_norm',
    'critic_grad_norm': f'{base}/high/critic_grad_norm/high/critic_grad_norm',
    'ratio': f'{base}/high/ratio/high/ratio',
}

for name, path in files.items():
    try:
        ea = EventAccumulator(path)
        ea.Reload()
        tags = ea.Tags()['scalars']
        for tag in tags:
            events = ea.Scalars(tag)
            n = len(events)
            print(f'=== {name} (tag={tag}, {n} points) ===')
            if n <= 20:
                for e in events:
                    print(f'  step={e.step:8d}  value={e.value:.6f}')
            else:
                for e in events[:5]:
                    print(f'  step={e.step:8d}  value={e.value:.6f}')
                print(f'  ...')
                mid = n // 4
                for e in events[mid-1:mid+2]:
                    print(f'  step={e.step:8d}  value={e.value:.6f}')
                print(f'  ... (25%) ...')
                mid = n // 2
                for e in events[mid-1:mid+2]:
                    print(f'  step={e.step:8d}  value={e.value:.6f}')
                print(f'  ... (50%) ...')
                mid = 3 * n // 4
                for e in events[mid-1:mid+2]:
                    print(f'  step={e.step:8d}  value={e.value:.6f}')
                print(f'  ... (75%) ...')
                for e in events[-5:]:
                    print(f'  step={e.step:8d}  value={e.value:.6f}')
            print()
    except Exception as ex:
        print(f'ERROR loading {name}: {ex}')
        print()
