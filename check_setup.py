"""
╔══════════════════════════════════════════════════════════╗
║        AnimaStudio — Setup Checker & Diagnostics        ║
║  Run this BEFORE starting the app to verify your setup  ║
╚══════════════════════════════════════════════════════════╝

Usage:
    python check_setup.py
"""

import os
import sys
import subprocess
import importlib

# ── Colour helpers (works on modern Windows 10+ terminals) ──────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

OK   = f"{GREEN}  ✔  OK{RESET}"
FAIL = f"{RED}  ✘  MISSING{RESET}"
WARN = f"{YELLOW}  ⚠  WARNING{RESET}"

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
SRC_PATH  = os.path.join(BASE_PATH, "src")

errors   = []
warnings = []

def header(title):
    print(f"\n{BOLD}{CYAN}{'─'*54}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*54}{RESET}")

def check(label, condition, hint=""):
    if condition:
        print(f"{OK}   {label}")
        return True
    else:
        print(f"{FAIL}   {label}")
        if hint:
            print(f"        {YELLOW}→ {hint}{RESET}")
        errors.append(label)
        return False

def warn(label, condition, hint=""):
    if condition:
        print(f"{OK}   {label}")
    else:
        print(f"{WARN}   {label}")
        if hint:
            print(f"        {YELLOW}→ {hint}{RESET}")
        warnings.append(label)


# ══════════════════════════════════════════════════════════════
# 1.  Python Version
# ══════════════════════════════════════════════════════════════
header("1. Python Version")
py = sys.version_info
label = f"Python {py.major}.{py.minor}.{py.micro}"
if py.major == 3 and py.minor >= 10:
    print(f"{OK}   {label}")
else:
    print(f"{WARN}   {label}  (recommend Python 3.10+)")
    warnings.append("Python version < 3.10")


# ══════════════════════════════════════════════════════════════
# 2.  Required Python Packages
# ══════════════════════════════════════════════════════════════
header("2. Python Packages  (pip install -r requirements.txt)")

PACKAGES = {
    "flask"               : "flask",
    "pandas"              : "pandas",
    "numpy"               : "numpy",
    "faiss"               : "faiss",          # faiss-cpu on PyPI
    "sentence_transformers": "sentence_transformers",
    "cv2"                 : "opencv-python",
    "moviepy"             : "moviepy",
    "PIL"                 : "Pillow",
    "nltk"                : "nltk",
    "torch"               : "torch",
}

for module, pip_name in PACKAGES.items():
    try:
        importlib.import_module(module)
        print(f"{OK}   {pip_name}")
    except ImportError:
        print(f"{FAIL}   {pip_name}")
        errors.append(f"Missing package: {pip_name}")
        print(f"        {YELLOW}→ Run:  pip install {pip_name}{RESET}")


# ══════════════════════════════════════════════════════════════
# 3.  FFmpeg
# ══════════════════════════════════════════════════════════════
header("3. FFmpeg  (required for video encoding)")

ffmpeg_ok = False
for candidate in ["ffmpeg", r"C:\ffmpeg\bin\ffmpeg.exe",
                  r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"]:
    try:
        result = subprocess.run(
            [candidate, "-version"],
            capture_output=True, timeout=5
        )
        if result.returncode == 0:
            version_line = result.stdout.decode(errors="ignore").split("\n")[0]
            print(f"{OK}   FFmpeg found  ({candidate})")
            print(f"        {CYAN}  {version_line[:60]}{RESET}")
            ffmpeg_ok = True
            break
    except Exception:
        pass

if not ffmpeg_ok:
    print(f"{FAIL}   FFmpeg not found in PATH or standard locations")
    print(f"        {YELLOW}→ Download: https://ffmpeg.org/download.html{RESET}")
    print(f"        {YELLOW}→ Add ffmpeg/bin to your System PATH{RESET}")
    errors.append("FFmpeg not found")


# ══════════════════════════════════════════════════════════════
# 4.  Critical Data Files (pre-built indexes)
# ══════════════════════════════════════════════════════════════
header("4. Pre-Built Index Files  (download from Google Drive link in README)")

data_files = {
    "combined_data.csv"        : os.path.join(SRC_PATH, "combined_data.csv"),
    "combined_embeddings.npy"  : os.path.join(SRC_PATH, "combined_embeddings.npy"),
    "combined_faiss.index"     : os.path.join(SRC_PATH, "combined_faiss.index"),
    "ucf101_captions.csv"      : os.path.join(SRC_PATH, "ucf101_captions.csv"),
}

for name, path in data_files.items():
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"{OK}   {name}  ({size_mb:.0f} MB)")
    else:
        print(f"{FAIL}   {name}")
        print(f"        {YELLOW}→ Download from the Google Drive link in README{RESET}")
        print(f"        {YELLOW}→ Place in:  {SRC_PATH}{RESET}")
        errors.append(f"Missing data file: {name}")


# ══════════════════════════════════════════════════════════════
# 5.  MSR-VTT Video Dataset
# ══════════════════════════════════════════════════════════════
header("5. MSR-VTT Video Dataset")

MSRVTT_VIDEOS = os.path.join(BASE_PATH, "datasets-project", "data",
                             "MSRVTT", "MSRVTT", "videos", "all")
MSRVTT_ANNOT  = os.path.join(BASE_PATH, "datasets-project", "data",
                             "MSRVTT", "MSRVTT", "annotation", "MSR_VTT.json")

if os.path.exists(MSRVTT_VIDEOS):
    clip_count = len([f for f in os.listdir(MSRVTT_VIDEOS) if f.endswith(".mp4")])
    print(f"{OK}   MSR-VTT videos folder  ({clip_count} mp4 files found)")
else:
    print(f"{WARN}   MSR-VTT videos folder not found")
    print(f"        {YELLOW}→ Expected: {MSRVTT_VIDEOS}{RESET}")
    print(f"        {YELLOW}→ Without videos, the app uses comic panel fallback (still works!){RESET}")
    warnings.append("MSR-VTT videos missing (fallback mode will be used)")

check("MSR-VTT annotation file  (MSR_VTT.json)",
      os.path.exists(MSRVTT_ANNOT),
      f"Place MSR_VTT.json at: {MSRVTT_ANNOT}")


# ══════════════════════════════════════════════════════════════
# 6.  UCF-101 Dataset (optional)
# ══════════════════════════════════════════════════════════════
header("6. UCF-101 Dataset  (optional — action category booster)")

UCF_PATH = os.path.join(BASE_PATH, "datasets-project", "UCF101", "UCF-101")
if os.path.exists(UCF_PATH):
    cats = [d for d in os.listdir(UCF_PATH)
            if os.path.isdir(os.path.join(UCF_PATH, d))]
    print(f"{OK}   UCF-101 dataset  ({len(cats)} categories found)")
else:
    print(f"{WARN}   UCF-101 not found  (optional — MSR-VTT alone is sufficient)")
    warnings.append("UCF-101 missing (optional)")


# ══════════════════════════════════════════════════════════════
# 7.  Ollama / Gemma3 (optional)
# ══════════════════════════════════════════════════════════════
header("7. Ollama + Gemma3 4B  (optional — enhances search quality)")

try:
    import urllib.request
    req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
    with urllib.request.urlopen(req, timeout=3) as resp:
        if resp.status == 200:
            print(f"{OK}   Ollama is running at http://localhost:11434")
            import json
            data = json.loads(resp.read().decode())
            model_names = [m.get("name","") for m in data.get("models",[])]
            gemma_found = any("gemma3" in n for n in model_names)
            if gemma_found:
                print(f"{OK}   gemma3:4b model is available")
            else:
                print(f"{WARN}   gemma3:4b not found  (run: ollama pull gemma3:4b)")
                warnings.append("gemma3:4b model not pulled")
        else:
            raise Exception()
except Exception:
    print(f"{WARN}   Ollama not running  (optional — app works fine without it)")
    print(f"        {YELLOW}→ Install: https://ollama.com/{RESET}")
    print(f"        {YELLOW}→ Then run: ollama run gemma3:4b{RESET}")
    warnings.append("Ollama not running (optional)")


# ══════════════════════════════════════════════════════════════
# 8.  Output Directories
# ══════════════════════════════════════════════════════════════
header("8. Required Directories")

for d in ["temp", "output"]:
    full = os.path.join(BASE_PATH, d)
    if not os.path.exists(full):
        os.makedirs(full)
        print(f"{OK}   {d}/  (created automatically)")
    else:
        print(f"{OK}   {d}/  (exists)")


# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
print(f"\n{BOLD}{'═'*54}{RESET}")
print(f"{BOLD}  SUMMARY{RESET}")
print(f"{BOLD}{'═'*54}{RESET}")

if not errors and not warnings:
    print(f"\n{GREEN}{BOLD}  🎉 Everything looks great! Run the app with:{RESET}")
    print(f"\n      {CYAN}python src/app.py{RESET}")
    print(f"\n  Then open:  {CYAN}http://127.0.0.1:5000{RESET}\n")

elif not errors:
    print(f"\n{YELLOW}  ⚠  {len(warnings)} optional warning(s) — app will still work:{RESET}")
    for w in warnings:
        print(f"      • {w}")
    print(f"\n{GREEN}{BOLD}  ✔  Run the app with:{RESET}")
    print(f"\n      {CYAN}python src/app.py{RESET}")
    print(f"\n  Then open:  {CYAN}http://127.0.0.1:5000{RESET}\n")

else:
    print(f"\n{RED}  ✘  {len(errors)} critical issue(s) must be fixed:{RESET}")
    for e in errors:
        print(f"      • {e}")
    if warnings:
        print(f"\n{YELLOW}  ⚠  {len(warnings)} optional warning(s):{RESET}")
        for w in warnings:
            print(f"      • {w}")
    print(f"\n  Fix the issues above, then run this script again.\n")

print(f"{BOLD}{'═'*54}{RESET}\n")
