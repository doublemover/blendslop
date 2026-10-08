"""Isolated approved CPU DVX helper. Pickle inputs are trusted local job packets."""
from pathlib import Path
import argparse
import json
import sys
import importlib.util
import pickle
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"blender_blocking")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--directory")
    parser.add_argument("--idle-timeout",type=float,default=60.)
    parser.add_argument("--qualify", action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--output")
    args = parser.parse_args()
    # Load only this numeric adapter; importing the reconstruction package
    # would pull unrelated image/Blender dependencies into the CPU helper.
    spec = importlib.util.spec_from_file_location("blendslop_dvx_helper",
        ROOT/"blender_blocking/reconstruction/differentiable/dvx_adapter.py")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    if args.serve:
        import os,time,traceback
        directory=Path(args.directory)
        ready=directory/'ready.json'
        temporary=directory/'ready.tmp'
        temporary.write_text(json.dumps({**adapter.dependency_state(),'pid':os.getpid()}))
        os.replace(temporary,ready)
        while not (directory/'stop').exists():
            heartbeat=directory/'heartbeat'
            if not heartbeat.exists() or time.time()-heartbeat.stat().st_mtime>args.idle_timeout:
                break
            jobs=sorted(directory.glob('*.input.pkl'))
            if not jobs:
                time.sleep(.01);continue
            for source in jobs:
                job=source.name[:-len('.input.pkl')]
                try:
                    with source.open('rb') as stream:payload=pickle.load(stream)
                    value=adapter.fit_job(payload)
                    response={'job':job,'status':'success','value':value}
                except Exception:
                    response={'job':job,'status':'failed','error':traceback.format_exc()}
                source.unlink(missing_ok=True)
                destination=directory/(job+'.output.pkl')
                temporary=destination.with_suffix('.tmp')
                with temporary.open('wb') as stream:pickle.dump(response,stream,protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(temporary,destination)
        return
    if args.probe:
        print(json.dumps(adapter.dependency_state()))
    elif args.qualify:
        result = adapter.tiny_gradient_check(approved=True)
        print(json.dumps(result))
        if not result["passed"]:
            raise RuntimeError("DVX gradient qualification failed")
    else:
        with Path(args.input).open("rb") as stream:
            payload = pickle.load(stream)
        result = adapter.fit_job(payload)
        with Path(args.output).open("wb") as stream:
            pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
