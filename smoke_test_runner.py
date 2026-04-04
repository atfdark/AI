"""Workspace-wide smoke test runner for launcher modes and test scripts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class SmokeResult:
    name: str
    command: list[str]
    status: str
    returncode: int | None
    duration_sec: float
    timeout_sec: int
    output_excerpt: str


def timeout_for_target(name: str) -> int:
    lowered = name.lower()
    if "ml_comparison" in lowered or "performance" in lowered:
        return 240
    if "launcher_comprehensive_test" in lowered:
        return 180
    if "microphone" in lowered:
        return 120
    return 90


def build_targets(py_exe: str, root: Path) -> list[tuple[str, list[str]]]:
    targets: list[tuple[str, list[str]]] = [
        ("launcher_status", [py_exe, "enhanced_launcher.py", "--status"]),
        ("launcher_agent_test", [py_exe, "enhanced_launcher.py", "--agent-test"]),
        ("launcher_comprehensive_test", [py_exe, "enhanced_launcher.py", "--test"]),
    ]

    for script in sorted(root.glob("test_*.py")):
        if script.name == "smoke_test_runner.py":
            continue
        targets.append((script.stem, [py_exe, script.name]))

    return targets


def run_target(name: str, command: list[str], cwd: Path, timeout_sec: int) -> SmokeResult:
    started = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            encoding="utf-8",
            errors="replace",
            check=False,
            env=env,
        )
        elapsed = time.perf_counter() - started
        joined = (proc.stdout or "") + ("\n" if proc.stderr else "") + (proc.stderr or "")
        excerpt = joined[-4000:]
        status = "pass" if proc.returncode == 0 else "fail"
        return SmokeResult(
            name=name,
            command=command,
            status=status,
            returncode=proc.returncode,
            duration_sec=round(elapsed, 2),
            timeout_sec=timeout_sec,
            output_excerpt=excerpt,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        joined = ((exc.stdout or "") + "\n" + (exc.stderr or "")).strip()
        return SmokeResult(
            name=name,
            command=command,
            status="timeout",
            returncode=None,
            duration_sec=round(elapsed, 2),
            timeout_sec=timeout_sec,
            output_excerpt=joined[-4000:],
        )
    except Exception as exc:  # defensive catch so one script never stops the suite
        elapsed = time.perf_counter() - started
        return SmokeResult(
            name=name,
            command=command,
            status="error",
            returncode=None,
            duration_sec=round(elapsed, 2),
            timeout_sec=timeout_sec,
            output_excerpt=str(exc),
        )


def write_reports(results: list[SmokeResult], report_dir: Path) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "smoke_test_report.json"
    md_path = report_dir / "smoke_test_report.md"

    summary = {
        "total": len(results),
        "passed": sum(1 for r in results if r.status == "pass"),
        "failed": sum(1 for r in results if r.status == "fail"),
        "timeouts": sum(1 for r in results if r.status == "timeout"),
        "errors": sum(1 for r in results if r.status == "error"),
    }

    payload = {
        "summary": summary,
        "results": [asdict(r) for r in results],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Smoke Test Report")
    lines.append("")
    lines.append(
        f"Total: {summary['total']} | Passed: {summary['passed']} | "
        f"Failed: {summary['failed']} | Timeouts: {summary['timeouts']} | Errors: {summary['errors']}"
    )
    lines.append("")
    lines.append("| Target | Status | Return Code | Duration (s) | Timeout (s) |")
    lines.append("|---|---|---:|---:|---:|")

    for r in results:
        rc = "-" if r.returncode is None else str(r.returncode)
        lines.append(f"| {r.name} | {r.status} | {rc} | {r.duration_sec} | {r.timeout_sec} |")

    lines.append("")
    lines.append("## Failures And Timeouts")
    lines.append("")
    problematic = [r for r in results if r.status in {"fail", "timeout", "error"}]
    if not problematic:
        lines.append("All targets passed.")
    else:
        for r in problematic:
            lines.append(f"### {r.name}")
            lines.append("")
            lines.append("```text")
            lines.append(r.output_excerpt or "(no captured output)")
            lines.append("```")
            lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> int:
    root = Path(__file__).resolve().parent
    py_exe = sys.executable
    targets = build_targets(py_exe, root)

    print(f"Running smoke tests for {len(targets)} targets...")
    results: list[SmokeResult] = []

    for index, (name, command) in enumerate(targets, start=1):
        timeout_sec = timeout_for_target(name)
        print(f"[{index}/{len(targets)}] {name} (timeout={timeout_sec}s)")
        result = run_target(name=name, command=command, cwd=root, timeout_sec=timeout_sec)
        results.append(result)
        print(f"  -> {result.status} ({result.duration_sec}s)")

    json_path, md_path = write_reports(results, root / "artifacts" / "reports")

    passed = sum(1 for r in results if r.status == "pass")
    print(f"\nSmoke test finished: {passed}/{len(results)} passed")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
