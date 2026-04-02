"""Print every 50th point for key curves to see full trajectory."""
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

base = 'c:/DCNLab/UCMEC/UCMEC-mmWave-Fronthaul/results/hotspotEnv/nlos_cluster/rmappo/hierarchical_hotspot_heuristic/run2/logs'

curves = {
    'high_reward': f'{base}/average_episode_rewards_high/average_episode_rewards_high',
    'dist_entropy': f'{base}/high/dist_entropy/high/dist_entropy',
    'value_loss': f'{base}/high/value_loss/high/value_loss',
    'policy_loss': f'{base}/high/policy_loss/high/policy_loss',
}

for name, path in curves.items():
    ea = EventAccumulator(path)
    ea.Reload()
    tag = ea.Tags()['scalars'][0]
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    vals = [e.value for e in events]

    print(f'=== {name} ({len(events)} points, step {steps[0]}-{steps[-1]}) ===')

    # Print every 50th point (every 100K steps)
    for i in range(0, len(events), 50):
        e = events[i]
        # also compute local mean (window of 10)
        lo = max(0, i-5)
        hi = min(len(vals), i+5)
        local_mean = np.mean(vals[lo:hi])
        print(f'  step={e.step:8d}  value={e.value:10.4f}  smooth={local_mean:10.4f}')
    # last point
    e = events[-1]
    lo = max(0, len(vals)-5)
    local_mean = np.mean(vals[lo:])
    print(f'  step={e.step:8d}  value={e.value:10.4f}  smooth={local_mean:10.4f}  (LAST)')

    # Summary stats by quarter
    q = len(vals) // 4
    for qi, label in enumerate(['Q1(0-25%)', 'Q2(25-50%)', 'Q3(50-75%)', 'Q4(75-100%)']):
        qvals = vals[qi*q:(qi+1)*q]
        print(f'  {label}: mean={np.mean(qvals):.4f}  std={np.std(qvals):.4f}  min={np.min(qvals):.4f}  max={np.max(qvals):.4f}')
    print()
