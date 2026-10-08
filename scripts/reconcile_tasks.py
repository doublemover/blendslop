"""Read-only inventory of every retained Markdown checkbox/spec section."""
from pathlib import Path
import csv
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "branch-audit-20261006"
OUT.mkdir(parents=True, exist_ok=True)
tracked = set(subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines())
rows, specs = [], []
sources = []
for scope in (ROOT / "docs", ROOT / "temp", ROOT / "blender_blocking"):
    for path in scope.rglob("*.md"):
        if path.is_symlink() or OUT in path.parents or "bounded-pass-20261006" in path.parts:
            continue
        sources.append(path)
for path in sources + [ROOT / "DONT_BREATHE_THIS.md", ROOT / "AGENTS.md", ROOT / "README.md"]:
    section = ""
    for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if line.startswith("#"):
            section = line.lstrip("# ")
            if "spec" in str(path).lower():
                specs.append({"source": str(path.relative_to(ROOT)), "line": number, "section": section})
        match = re.match(r"\s*[-*]\s+\[([ xX])\]\s*(.*)", line)
        todo = re.search(r"\b(TODO|FIXME|Still broken)\b", line)
        if not match and not todo:
            continue
        text = match[2] if match else line.strip()
        refs = re.findall(r"`([^`]+\.(?:py|md|json|yml))(?::\d+(?:-\d+)?)?`", text)
        existing = [p for p in refs if (ROOT / p.replace("\\", "/")).is_file()]
        rows.append({"source": str(path.relative_to(ROOT)), "line": number, "section": section,
                     "recorded_state": "recorded_done" if match and match[1].lower() == "x" else "recorded_open",
                     "verification": "historical_claim_requires_current_evidence",
                     "task": text, "existing_referenced_files": ";".join(existing),
                     "tracked_source": str(path.relative_to(ROOT)).replace("\\", "/") in tracked})
for name, data in (("task-records", rows), ("spec-sections", specs)):
    (OUT / f"{name}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    with (OUT / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
print(json.dumps({"task_records": len(rows), "recorded_open": sum(r["recorded_state"] == "recorded_open" for r in rows),
                  "spec_sections": len(specs), "markdown_sources": len(sources)}))
