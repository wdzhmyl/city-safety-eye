# 源代码备份归档（backups/）

本目录用于把**当前项目的源代码快照**按时间版本化留存进 git，便于日后回溯任意历史版本。

## 约定

- 每次备份生成**独立目录**，互不覆盖：`backups/<时间戳>/` 或 `backups/<时间戳>__<标签>/`
- 每个快照内含：
  - `manifest.txt` —— 备份时间、git 提交号、git 状态、文件清单、备注
  - `src/` —— 与项目同构的源码副本
- 自动排除：**模型权重(\*.pt)、Python 缓存、`.env`、`.gradio/`、venv、legacy** 等大文件/敏感/生成物，只留可重建项目的源码，避免仓库膨胀。

## 使用

```bash
# 生成并自动提交一份快照
python tools/backup_source.py

# 带标签（推荐，方便区分用途）
python tools/backup_source.py --name 大赛初版
python tools/backup_source.py --name 视频分析版

# 备注会写进 manifest.txt
python tools/backup_source.py --note "修复画框越界"

# 只想先生成、稍后手动提交
python tools/backup_source.py --no-commit
```

提交身份默认用 `xiaobai / 1140431160@qq.com`；如需改动可设环境变量：

```bash
GIT_USER_NAME=你的名 GIT_USER_EMAIL=你@邮箱.com python tools/backup_source.py
```

## 恢复 / 取用某份备份

```bash
# 查看某快照内容
ls backups/2026-10-09_14-30-00/src

# 整体还原到某快照（会覆盖当前源码，谨慎）
cp -r backups/2026-10-09_14-30-00/src/* .

# 用 git 直接回退到某次备份提交
git log --oneline -- backups/        # 找到对应 backup: ... 提交
git checkout <commit> -- backups/<时间戳>/src
```

## 注意

- 备份的是**源码快照**，不是运行环境（不含权重/缓存）。还原后仍需 `pip install -r requirements.txt`，首次检测会自动下载 `yolov8n.pt`。
- 历史快照会随 git 永久留存，仓库体积会随时间增长，属正常版本化代价。
