"""Offline paired masks/hulls and annotated ROI; no online consumers."""
from pathlib import Path
import json
import numpy as np


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path('outputs/module1_final_validation_refinement');dest=root/'critical_event_overlays'
    events=json.loads((root/'critical_event_root_cause.json').read_text())['events']
    rows=[json.loads(s) for s in (root/'replay/pixels.jsonl').read_text().splitlines()]
    lookup={(s['frame'],s['camera']):s for s in rows if s['method']=='final'}
    for index,event in enumerate(events):
        base=np.load(event['arrays']);final=np.load(dest/f'final_{event["frame"]}_{event["camera"]}.npz')
        fig,axes=plt.subplots(2,4,figsize=(16,8))
        layers=[('Current image',base['image']),('Old candidate',base['reuse_candidate_mask']),
            ('Final candidate',final['candidate']),('Final Dirty',final['dirty']),('Final Unknown',final['unknown']),
            ('Measured original change',base['raw_changed_mask']),('Online measured risk',base['measured_change_risk_mask']),
            ('Motion risk Dirty',final['motion_dirty'])]
        x0,y0,x1,y1=event['region'];h,w=base['image'].shape[:2]
        for ax,(name,value) in zip(axes.ravel(),layers):
            ax.imshow(value,cmap=None if value.ndim==3 else 'gray');ax.set_title(name,fontsize=9)
            ax.add_patch(plt.Rectangle((x0,y0),x1-x0,y1-y0,fill=False,color='red',linewidth=1))
            for envelope in lookup[(event['frame'],event['camera'])].get('motion',{}).get('envelopes',[]):
                p=np.asarray(envelope['hull']);ax.plot(*np.r_[p,p[:1]].T,color='cyan',linewidth=.5)
            ax.set_xlim(max(0,x0-40),min(w,x1+40));ax.set_ylim(min(h,y1+40),max(0,y0-40))
        fig.suptitle(f'{event["camera"]} {event["semantic_class"]}: red=offline GT, cyan=online historical motion hull')
        fig.tight_layout();fig.savefig(dest/f'paired_{index:02d}_{event["frame"]}_{event["camera"]}.png',dpi=130);plt.close(fig)


if __name__=='__main__':main()
