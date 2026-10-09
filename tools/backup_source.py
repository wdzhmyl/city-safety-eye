#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把当前项目的源代码快照备份到 backups/ 下，便于在 git 中版本化留存。

设计目标
--------
- 每次备份生成一个独立、带时间戳的目录，互不覆盖，可长期并存多份归档。
- 自动排除大文件/敏感/生成物（模型权重、缓存、.env、legacy、venv 等），
  只保留可重建项目的源码，避免仓库被权重、缓存撑爆。
- 生成 manifest.txt 记录元信息（时间 / git 提交号 / 文件清单），方便日后溯源。
- 默认直接 git 提交；用 --no-commit 可只生成不提交。

用法
----
    python tools/backup_source.py                      # 生成并自动提交
    python tools/backup_source.py --name 大赛初版       # 带标签命名
    python tools/backup_source.py --note "修复画框bug"   # 备注写进 manifest
    python tools/backup_source.py --no-commit          # 只生成，不提交

快照目录结构
------------
    backups/
        2026-10-09_14-30-00/        (或 2026-10-09_14-30-00__大赛初版/)
            manifest.txt            # 元信息 + 文件清单
            src/                    # 源码副本（与项目同构）
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKUP_ROOT = os.path.join(ROOT, "backups")

# 不纳入备份的内容（大文件 / 缓存 / 敏感 / 生成物 / 旧版）
EXCLUDE_DIRS = {
    ".git", "__pycache__", "legacy", "backups", "node_modules",
    "venv", ".venv", "env", "tmp", ".gradio", ".idea", ".vscode",
    ".pytest_cache", ".mypy_cache",
}
EXCLUDE_EXT = {".pyc", ".pyo", ".pt", ".onnx", ".engine", ".pth"}
EXCLUDE_FILES = {".env", ".DS_Store", "Thumbs.db", ".coverage"}


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return ""


def _iter_source_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        rel = os.path.relpath(dirpath, ROOT)
        parts = [p for p in rel.split(os.sep) if p]
        if any(p in EXCLUDE_DIRS for p in parts):
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            if fn in EXCLUDE_FILES or os.path.splitext(fn)[1].lower() in EXCLUDE_EXT:
                continue
            yield os.path.join(dirpath, fn)


def main() -> int:
    ap = argparse.ArgumentParser(description="把源代码快照备份进仓库的 backups/ 目录")
    ap.add_argument("--name", default="", help="备份标签，会拼到目录名末尾")
    ap.add_argument("--note", default="", help="备份说明，写入 manifest.txt")
    ap.add_argument("--no-commit", action="store_true", help="只生成快照，不执行 git 提交")
    args = ap.parse_args()

    stamp = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_name = stamp if not args.name else f"{stamp}__{args.name}"
    dest = os.path.join(BACKUP_ROOT, dir_name)
    src_dest = os.path.join(dest, "src")
    os.makedirs(src_dest, exist_ok=True)

    files = sorted(_iter_source_files())
    manifest = [
        f"backup_time : {stamp}",
        f"name        : {args.name}",
        f"note        : {args.note}",
        f"git_commit  : {_git('rev-parse', 'HEAD') or '(unknown)'}",
        f"git_status  : {_git('status', '--porcelain') or '(clean)'}",
        f"file_count  : {len(files)}",
        "",
        "files:",
    ]

    for f in files:
        rel = os.path.relpath(f, ROOT)
        target = os.path.join(src_dest, rel)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(f, target)
        manifest.append(f"  {rel}")

    with open(os.path.join(dest, "manifest.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(manifest) + "\n")

    rel_dest = os.path.relpath(dest, ROOT)
    print(f"[备份] 已复制 {len(files)} 个文件 -> {rel_dest}")

    if args.no_commit:
        print(f"[提示] 未提交。需要时可执行:\n"
              f"        git add {rel_dest} && git commit -m \"backup: {dir_name}\"")
        return 0

    name = os.environ.get("GIT_USER_NAME", "xiaobai")
    email = os.environ.get("GIT_USER_EMAIL", "1140431160@qq.com")
    try:
        subprocess.check_call(["git", "add", rel_dest], cwd=ROOT)
        # 同时跟踪 manifest 之外可能的目录级 .gitkeep 变更
        subprocess.check_call(["git", "add", "-A", "--renormalize"], cwd=ROOT)
        msg = f"backup: source snapshot {dir_name}"
        if args.note:
            msg += f" ({args.note})"
        subprocess.check_call(
            ["git", "-c", f"user.name={name}", "-c", f"user.email={email}",
             "commit", "-q", "-m", msg],
            cwd=ROOT,
        )
        print(f"[提交] 已提交到 git: {msg}")
    except subprocess.CalledProcessError:
        print("[提交] git 提交失败，请检查仓库状态后手动提交。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
