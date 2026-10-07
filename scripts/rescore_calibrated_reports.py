"""Independent raw pixel rescore of preserved calibrated renders; solver never rerun."""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'blender_blocking')]
from blender_blocking.verify_setup import configure_dependency_paths
configure_dependency_paths()
from PIL import Image
import numpy as np
from blender_blocking.evaluation.silhouette_eval import evaluate_silhouette_pair,SilhouetteGateConfig

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',default='camera_control');a=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else None)
    data=json.loads((a.root/(a.arm+'.json')).read_text())
    for row in data['rows']:
        if not row.get('result_path'):continue
        payload=json.loads(Path(row['result_path']).read_text());manifest=json.loads((a.root/'references'/row['case']/'manifest.json').read_text())
        views={}
        for v in ['front','side','top']:
            ref=np.asarray(Image.open(manifest['views'][v]).convert('RGBA'))[:,:,3]>127
            pred=np.asarray(Image.open(payload['rendered_paths'][v]).convert('RGBA'))[:,:,3]>127
            views[v]=evaluate_silhouette_pair(ref,pred,view=v,config=SilhouetteGateConfig(min_area_iou=.7,min_boundary_iou=.1,max_signed_distance_loss=.1))
        row['workflow_reported_metrics']={'average_iou':row['average_iou'],'min_view_iou':row['min_view_iou'],'passed':row['passed'],'views':row['views']}
        row.update(views=views,average_iou=float(np.mean([v['area_iou'] for v in views.values()])),min_view_iou=min(v['area_iou'] for v in views.values()),passed=all(v['passed'] for v in views.values()))
        row['silhouette_protocol']='raw alpha >127; no hole filling, component removal, crop or independent normalization'
    out=a.root/(a.arm+'-raw.json');out.write_text(json.dumps(data,indent=2)); print([(r['case'],r['requested_mode'],round(r.get('average_iou',0),3),r.get('passed')) for r in data['rows']])
if __name__=='__main__':main()
