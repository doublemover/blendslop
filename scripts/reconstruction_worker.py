"""Entry point for the persistent Blender/Python reconstruction worker."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "blender_blocking")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worker", required=True)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)
    from blender_blocking.verify_setup import configure_dependency_paths
    configure_dependency_paths()
    from blender_blocking.reconstruction.process_executor import worker_main
    worker_main(args.root, args.worker)


if __name__ == "__main__":
    main()
