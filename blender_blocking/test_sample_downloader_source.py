"""Source contracts and optional no-network PowerShell checks for sample downloads.

Python is only the repository's test harness; the downloader is self-contained
PowerShell/.NET. Source/path-model checks are not PowerShell execution evidence.
"""
from __future__ import annotations

import json
import ntpath
from pathlib import Path
import posixpath
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "Get-BlendslopSamples.ps1"
PWSH = shutil.which("pwsh")


class TestSampleDownloaderSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SCRIPT.read_text(encoding="utf-8")
        match = re.search(r"\$ManifestJson = @'\n(.*?)\n'@", cls.source, re.S)
        if match is None:
            raise AssertionError("The self-contained manifest is missing")
        cls.manifest = json.loads(match.group(1))

    def test_default_is_script_relative_and_has_no_prompt(self):
        self.assertIn(
            "$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))",
            self.source,
        )
        self.assertIn(
            "$Folder = [IO.Path]::Combine($repositoryRoot, 'temp', 'sample-pack')",
            self.source,
        )
        self.assertNotIn("Read-Host", self.source)
        self.assertNotRegex(self.source, r"(?i)\$PWD\b|Get-Location")
        self.assertIn("[Alias('Destination')]", self.source)
        self.assertIn("[string] $Folder", self.source)

    def test_script_relative_path_model_is_cwd_independent(self):
        # Independent platform path model of the checked two source expressions.
        # Actual PowerShell evaluation is tested separately when pwsh is present.
        fixtures = (
            (posixpath, "/repo with spaces/scripts", "/repo with spaces/temp/sample-pack", ("/", "/tmp/elsewhere")),
            (ntpath, r"C:\repo with spaces\scripts", r"C:\repo with spaces\temp\sample-pack", ("D:\\", r"C:\other")),
        )
        for path_api, script_root, expected, caller_directories in fixtures:
            for caller_cwd in caller_directories:
                with self.subTest(script_root=script_root, caller_cwd=caller_cwd):
                    repo_root = path_api.normpath(path_api.join(script_root, ".."))
                    self.assertEqual(path_api.join(repo_root, "temp", "sample-pack"), expected)

    def test_embedded_cohorts_and_sizes(self):
        meshes = self.manifest["partObjaverse"]["objects"]
        points = self.manifest["primitiveAnything"]["files"]
        outputs = self.manifest["superFit"]["files"]
        self.assertEqual(len(meshes), 16)
        self.assertEqual(len({row["id"] for row in meshes}), 16)
        self.assertEqual(sum(row["fileName"].endswith(".ply") for row in points), 12)
        self.assertEqual(sum(row["fileName"].endswith("/primitive_assembly.pkl") for row in outputs), 16)
        self.assertEqual(len(meshes) + len(points) + len(outputs), 65)
        payload = sum(row["compressed"] for row in meshes)
        payload += sum(row["expectedBytes"] for row in points + outputs)
        self.assertEqual(payload, 36_246_898)
        for row in meshes:
            self.assertEqual(row["name"], f"PartObjaverse-Tiny_mesh/{row['id']}.glb")
            self.assertGreater(row["size"], 0)
            self.assertLess(row["offset"] + row["compressed"], self.manifest["partObjaverse"]["archiveBytes"])
        source_ids = {row["id"] for row in meshes}
        fitted_ids = {row["fileName"].split("/")[-2] for row in outputs if row["fileName"].endswith("/primitive_assembly.pkl")}
        self.assertEqual(source_ids, fitted_ids)

    def test_runtime_dependencies_and_guards_remain_embedded(self):
        self.assertNotRegex(self.source, r"(?im)^\s*(?:&\s*)?(?:python\d*|py|pip|dotnet|Import-Module)\b")
        for guard in (
            "Expected HTTP $wanted, received $status. No whole-archive fallback.",
            "Content-Range does not exactly match the pinned archive and request.",
            "new DeflateStream(member, CompressionMode.Decompress, false)",
            "Pinned CRC32 mismatch.",
            "Pinned SHA256 mismatch; existing file preserved.",
            "[IO.File]::Move($stage,$path) # Never replaces an existing file.",
            "Stored/deflated ZIP member self-test failed.",
        ):
            self.assertIn(guard, self.source)
        main = self.source.split("# Normalize the three embedded manifests", 1)[1]
        self.assertLess(main.index("$PSCmdlet.ShouldProcess"), main.index("Add-Type"))
        self.assertLess(main.index("$PSCmdlet.ShouldProcess"), main.index("Ensure-Directory $script:Root"))

    @unittest.skipUnless(PWSH, "PowerShell unavailable; no parser/runtime claim")
    def test_powershell_whatif_default_from_another_directory(self):
        with tempfile.TemporaryDirectory() as caller:
            result = subprocess.run(
                [PWSH, "-NoProfile", "-File", str(SCRIPT), "-WhatIf"],
                cwd=caller, capture_output=True, text=True, timeout=30, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(f"Folder: {ROOT / 'temp' / 'sample-pack'}", result.stdout)
            self.assertEqual(list(Path(caller).iterdir()), [])

    @unittest.skipUnless(PWSH, "PowerShell unavailable; no parser/runtime claim")
    def test_powershell_whatif_honors_folder_override(self):
        with tempfile.TemporaryDirectory() as caller:
            target = Path(caller) / "custom sample folder"
            result = subprocess.run(
                [PWSH, "-NoProfile", "-File", str(SCRIPT), "-Folder", str(target), "-WhatIf"],
                cwd=caller, capture_output=True, text=True, timeout=30, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(f"Folder: {target}", result.stdout)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
