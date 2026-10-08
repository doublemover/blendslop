"""
Quick validation test - subset of objects to verify test suite works.

Tests 3 objects from different categories:
- cube (simple)
- vase (organic)
- table (furniture)

Usage:
    /Applications/Blender.app/Contents/MacOS/Blender --background --python test_suite_quick.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the full test suite
from test_suite_multiview import test_object, TEST_OBJECTS
import json
import numpy as np

# Subset for quick validation
QUICK_TEST_OBJECTS = [
    TEST_OBJECTS[0],  # cube (simple)
    TEST_OBJECTS[5],  # vase (organic)
    TEST_OBJECTS[9],  # table (furniture)
]


def main() -> None:
    """Run quick validation on 3 objects."""
    print("\n" + "=" * 70)
    print("QUICK VALIDATION TEST - 3 Objects")
    print("=" * 70)
    print(f"\nObjects: {[obj['name'] for obj in QUICK_TEST_OBJECTS]}")

    all_results = []

    for i, obj_config in enumerate(QUICK_TEST_OBJECTS, 1):
        print(f"\n[{i}/{len(QUICK_TEST_OBJECTS)}]")
        try:
            results = test_object(obj_config, resolution=128)
            all_results.append(results)
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            import traceback

            traceback.print_exc()
            all_results.append(
                {
                    "name": obj_config["name"],
                    "category": obj_config["category"],
                    "error": str(e),
                }
            )

    # Save results
    results_path = Path(__file__).resolve().parents[1] / "temp" / "suite" / "quick_test_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("QUICK TEST SUMMARY")
    print("=" * 70)

    successful = [r for r in all_results if "error" not in r]
    failed = [r for r in all_results if "error" in r]

    print(f"\nCompleted: {len(successful)}/{len(QUICK_TEST_OBJECTS)}")
    print(f"Failed: {len(failed)}")

    if successful:
        avg_3view = np.mean([r["3view_iou"] for r in successful])
        avg_12view = np.mean([r["12view_iou"] for r in successful])
        avg_improvement = np.mean([r["iou_improvement"] for r in successful])

        print(f"\nAverage IoU:")
        print(f"  3-view:  {avg_3view:.4f}")
        print(f"  12-view: {avg_12view:.4f}")
        print(
            f"  Improvement: {avg_improvement:+.4f} ({avg_improvement/avg_3view*100:+.1f}%)"
        )

        print(f"\nResults:")
        for r in successful:
            print(
                f"  {r['name']:15} - 3view: {r['3view_iou']:.4f}, 12view: {r['12view_iou']:.4f}, gain: {r['iou_improvement']:+.4f}"
            )

    if failed:
        print(f"\nFailed objects:")
        for r in failed:
            print(f"  {r['name']}: {r.get('error', 'Unknown error')}")

    print(f"\nResults saved to: {results_path}")

    # Validation
    if len(successful) == len(QUICK_TEST_OBJECTS):
        print("\n✓ VALIDATION SUCCESSFUL - Test suite ready for full run")
    else:
        print("\n⚠ VALIDATION FAILED - Fix errors before full run")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
