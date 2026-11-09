"""Entry point for the hosted demo.

Streamlit Community Cloud installs the requirements.txt next to this file, which leaves out the
heavy model libraries, so the app runs in its lightweight mode: BM25 retrieval, the lexical
truth highlighter, and extractive answers unless a visitor pastes their own Anthropic API key.
"""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("CITEBENCH_DEMO", "1")

runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
