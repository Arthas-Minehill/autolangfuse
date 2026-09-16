"""数据回收桌面入口。

推荐运行方式：
    python 一键启动数据回收.py

如果 Windows 已把 .py 关联到 Python，也可以直接双击本文件。
"""
from __future__ import annotations

import sys
import traceback
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPT_DIR = ROOT / "scripts"
LOG_PATH = ROOT / "logs" / "session_recovery_launcher.log"


def show_error(message: str) -> None:
    """即使从资源管理器启动，也用弹窗保留错误。"""
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("数据回收启动失败", message)
        root.destroy()
    except Exception:
        print(message, file=sys.stderr)


def main() -> int:
    try:
        sys.path.insert(0, str(SCRIPT_DIR))
        import session_recovery_launcher

        app = session_recovery_launcher.RecoveryApp()
        app.run()
        return 0
    except Exception as exc:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        detail = traceback.format_exc()
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[{datetime.now().isoformat()}] 启动失败\n{detail}\n")
        show_error(f"{exc}\n\n详细信息已写入：\n{LOG_PATH}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
