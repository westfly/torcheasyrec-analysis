#!/usr/bin/env bash
# 更新两个 git 子模块（torcheasyrec、external/recsys-examples）。
#
# 用法:
#   ./scripts/update_submodule.sh            # 同步到当前 .gitmodules 锁定的 commit
#   ./scripts/update_submodule.sh --latest   # 升级到上游 main 分支最新 commit
#
# 升级到 --latest 后请重新核对文档中受影响的源码行号引用。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LATEST=false
if [[ "${1:-}" == "--latest" ]]; then
  LATEST=true
fi

# 仓库相对路径数组
SUBMODULES=(
  "torcheasyrec"
  "external/recsys-examples"
)

update_one() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "==> [skip] $path 不存在（请先 git submodule update --init）"
    return
  fi

  echo "==> Updating submodule: $path"
  pushd "$path" >/dev/null

  if $LATEST; then
    # 升级到上游 main 分支最新 commit
    local branch
    branch="$(git rev-parse --abbrev-ref HEAD)"
    git fetch origin
    git reset --hard "origin/${branch}"
    echo "    -> reset to origin/${branch} @ $(git rev-parse --short HEAD)"
  else
    # 同步到 .gitmodules 锁定的 commit
    git fetch origin
    git checkout -
    echo "    -> checked out $(git rev-parse --short HEAD)"
  fi

  popd >/dev/null
}

for sub in "${SUBMODULES[@]}"; do
  update_one "$sub"
done

# 把子模块的 commit 引用写回主仓库
git add torcheasyrec external/recsys-examples 2>/dev/null || true

if git diff --cached --quiet; then
  echo "==> 子模块已是最新的，无需提交。"
else
  echo "==> 已暂存子模块 commit 变化。请确认后提交:"
  echo "    git diff --cached"
  echo "    git commit -m 'chore: bump submodules'"
fi
