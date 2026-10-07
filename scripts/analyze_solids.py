"""Saved Blender fixtures and candidate meshes checked with installed Open3D CPU geometry."""
from pathlib import Path
import argparse,hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'blender_blocking')]
from blender_blocking.evaluation.comparable_geometry import read_obj
from blender_blocking.evaluation.solid_validity import solid_validity_report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--phase',type=Path,required=True);args=parser.parse_args()
    import open3d as o3d
    items=[]
    fixture=args.phase/'solids'/'result.json'
    if fixture.exists():
        for row in json.loads(fixture.read_text())['rows']:
            if row.get('mesh'):items.append((row['case'],Path(row['mesh'])))
    for source in ['paired','heldout']:
        root=args.phase/source
        for arm in ['camera_control','mask_control','improved','final','poisson','poisson_closed']:
            index=root/(arm+'.json')
            if not index.exists():continue
            for row in json.loads(index.read_text())['rows']:
                file=Path(row.get('result_path',''))
                if not file.is_file():continue
                payload=json.loads(file.read_text())
                path=file.parent/'artifacts'/'validated-mesh.obj'
                if path.is_file():items.append((f'{source}/{arm}/{row["case"]}/{row["requested_mode"]}',path))
    rows=[]
    cache={}
    for case,path in items:
        started=time.perf_counter()
        try:
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if digest not in cache:
                cache[digest]=solid_validity_report(*read_obj(path),native_intersections=True)
            validity=cache[digest]
            rows.append({'case':case,'mesh':str(path),'validity':validity,'elapsed_s':time.perf_counter()-started})
        except Exception as exc:rows.append({'case':case,'mesh':str(path),'error':str(exc)})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'open3d':o3d.__version__, 'python':sys.version,
            'complete':len(rows)==len(items), 'expected_records':len(items), 'rows':rows},indent=2))
        print(f'Checked {len(rows)}/{len(items)}: {case}', flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'open3d':o3d.__version__,'python':sys.version,'complete':True,'expected_records':len(items),'rows':rows},indent=2))
    print(f'Checked {len(rows)} solid records')
if __name__=='__main__':main()
