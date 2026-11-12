import os
import re
import ast
import yaml
from pathlib import Path
import pytest
from markdown_it import MarkdownIt

SKILLS_DIR = Path("C:/Users/yashb_f1ls/.gemini/config/skills")
TARGET_SKILLS = [
    "creative-studio-design",
    "frontier-deep-learning-engineering",
    "agentic-developer-workflow",
    "host-app-viewer",
]


@pytest.mark.parametrize("skill_name", TARGET_SKILLS)
def test_skill_frontmatter_and_markdown(skill_name):
    skill_path = SKILLS_DIR / skill_name
    assert skill_path.is_dir(), f"Skill directory {skill_path} missing"

    skill_file = skill_path / "SKILL.md"
    assert skill_file.is_file(), f"SKILL.md missing in {skill_path}"

    content = skill_file.read_text(encoding="utf-8")
    assert content.startswith("---"), "Frontmatter must begin with ---"

    parts = content.split("---", 2)
    assert len(parts) >= 3, "Frontmatter must be enclosed by --- delimiters"

    meta = yaml.safe_load(parts[1])
    assert isinstance(meta, dict), "Frontmatter must parse into a dict"
    assert "name" in meta and meta["name"], "Frontmatter missing 'name'"
    assert "description" in meta and meta["description"], "Frontmatter missing 'description'"

    # Validate markdown parsing
    md = MarkdownIt()
    tokens = md.parse(parts[2])
    assert len(tokens) > 10, "Markdown body should have substantial content"


def test_agentic_workflow_scripts_execution():
    script_dir = SKILLS_DIR / "agentic-developer-workflow" / "scripts"
    telemetry_script = script_dir / "benchmark_telemetry.py"
    invariants_script = script_dir / "verify_invariants.py"

    assert telemetry_script.is_file()
    assert invariants_script.is_file()

    # Verify Python syntax
    ast.parse(telemetry_script.read_text(encoding="utf-8"))
    ast.parse(invariants_script.read_text(encoding="utf-8"))


def test_frontier_examples_python():
    nvme_script = (
        SKILLS_DIR / "frontier-deep-learning-engineering" / "examples" / "nvme_streamer.py"
    )
    assert nvme_script.is_file()
    ast.parse(nvme_script.read_text(encoding="utf-8"))


def test_host_app_viewer_scripts():
    scripts_dir = SKILLS_DIR / "host-app-viewer" / "scripts"
    list_script = scripts_dir / "list_windows.ps1"
    capture_script = scripts_dir / "capture_window.ps1"
    interact_script = scripts_dir / "interact_app.ps1"

    assert list_script.is_file(), "list_windows.ps1 missing"
    assert capture_script.is_file(), "capture_window.ps1 missing"
    assert interact_script.is_file(), "interact_app.ps1 missing"

    import subprocess
    cmd_list = [
        "powershell",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(list_script),
        "-AsJson",
    ]
    res_list = subprocess.run(cmd_list, capture_output=True, text=True, check=True)
    assert res_list.returncode == 0
    assert len(res_list.stdout.strip()) > 0
