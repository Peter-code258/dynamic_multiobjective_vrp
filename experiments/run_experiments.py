"""实验运行快捷入口。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["run", "--config", "config/quick.yaml"]))
