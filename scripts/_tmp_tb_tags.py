import os
from tensorboard.backend.event_processing import event_accumulator
root='results/hotspotEnv/nlos_cluster/rmappo/hierarchical_hotspot_heuristic_noattn_lp/run1/logs'
for d,_,fs in os.walk(root):
    for f in fs:
        if 'tfevents' not in f:
            continue
        p=os.path.join(d,f)
        ea=event_accumulator.EventAccumulator(p,size_guidance={event_accumulator.SCALARS:0,event_accumulator.TENSORS:0})
        ea.Reload()
        tags=ea.Tags()
        print('FILE',p)
        for k,v in tags.items():
            if v:
                print(' ',k,len(v))
                print('   sample',v[:5])
