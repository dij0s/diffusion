import os
import runpy
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path("/content/proj")

# unpack the freshly uploaded code (overwrites on every run)
with zipfile.ZipFile("/content/proj.zip") as z:
    z.extractall(ROOT)

# install the dependencies declared in pyproject.toml (the project itself is
# imported from src/ via sys.path, so it isn't installed)
if shutil.which("uv"):
    cmd = ["uv", "pip", "install", "--system", "-q", "-r", str(ROOT / "pyproject.toml")]
else:
    cmd = [sys.executable, "-m", "pip", "install", "-q", str(ROOT)]
subprocess.run(cmd, check=True)

# emulate `uv run --env-file .env`
for line in (ROOT / ".env").read_text().splitlines():
    line = line.strip().removeprefix("export ")
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    os.environ[key.strip()] = value.strip().strip("'\"")

# the kernel is persistent: drop stale modules so re-runs use new code
for name in [m for m in sys.modules if m == "diffusion" or m.startswith("diffusion.")]:
    del sys.modules[name]

sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)  # so root_dir="data" resolves to /content/proj/data

# emulate: scripts/train_model.py config.yaml --log
sys.argv = ["scripts/train_model.py", "config.yaml", "--log"]
runpy.run_path("scripts/train_model.py", run_name="__main__")
