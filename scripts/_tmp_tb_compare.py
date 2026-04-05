import os
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

runs = {
    'noattn_lp_run1':'results/hotspotEnv/nlos_cluster/rmappo/hierarchical_hotspot_heuristic_noattn_lp/run1/logs',
    'set_run5':'results/hotspotEnv/nlos_cluster/rmappo/hierarchical_hotspot_heuristic/run5/logs',
}

def collect_scalar(root, tag):
    vals=[]
    for d,_,fs in os.walk(root):
        for f in fs:
            if 'tfevents' not in f:
                continue
            p=os.path.join(d,f)
            ea=event_accumulator.EventAccumulator(p, size_guidance={event_accumulator.SCALARS:0})
            ea.Reload()
            if tag in ea.Tags().get('scalars',[]):
                vals.extend([(e.step, e.value) for e in ea.Scalars(tag)])
    vals=sorted(vals, key=lambda x:x[0])
    if not vals:
        return np.array([]), np.array([])
    ded={}
    for s,v in vals:
        ded[s]=v
    steps=np.array(sorted(ded.keys()),dtype=np.int64)
    arr=np.array([ded[s] for s in steps],dtype=np.float64)
    return steps, arr

TAGS=[
    'average_episode_rewards_high/average_episode_rewards_high',
    'average_episode_rewards/average_episode_rewards',
    'high/dist_entropy/high/dist_entropy',
    'high/value_loss/high/value_loss',
    'high/policy_loss/high/policy_loss',
]

for tag in TAGS:
    print(f'\\n=== {tag} ===')
    out={}
    for name,root in runs.items():
        s,v=collect_scalar(root,tag)
        out[name]=(s,v)
        print(name, 'n=',len(v), 'step_end=', int(s[-1]) if len(s) else -1, 'last=', float(v[-1]) if len(v) else None)
    a=out['noattn_lp_run1'][1]
    b=out['set_run5'][1]
    n=min(len(a),len(b))
    if n==0:
        continue
    a=a[:n]; b=b[:n]
    print('common_n=',n)
    windows=[(0,20,'1-20'),(20,40,'21-40'),(40,60,'41-60'),(60,80,'61-80'),(80,100,'81-100')]
    for i,j,label in windows:
        if n>=j:
            ma=float(np.mean(a[i:j]))
            mb=float(np.mean(b[i:j]))
            print(label, 'noattn=', round(ma,4), 'set=', round(mb,4), 'diff=', round(ma-mb,4))
