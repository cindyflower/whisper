#!/usr/bin/env bash
# 本機 Whisper UI；固定加上 NO_PROXY，避免系統代理導致無法連線 127.0.0.1
set -euo pipefail
cd "$(dirname "$0")"
export NO_PROXY="127.0.0.1,localhost,0.0.0.0,${NO_PROXY:-}"
export no_proxy="$NO_PROXY"

# 若使用者未指定埠，從 7860 起自動找可用埠，避免啟動失敗。
if [[ -z "${GRADIO_SERVER_PORT:-}" ]]; then
  port=7860
  max_port=7999
  while lsof -iTCP:"${port}" -sTCP:LISTEN -n -P >/dev/null 2>&1; do
    ((port++))
    if ((port > max_port)); then
      echo "找不到可用埠（已嘗試 7860-${max_port}），請手動指定 GRADIO_SERVER_PORT" >&2
      exit 1
    fi
  done
  export GRADIO_SERVER_PORT="${port}"
  echo "自動選擇可用埠：${GRADIO_SERVER_PORT}"
fi

# shellcheck disable=SC1091
source .venv/bin/activate
exec python -u local_ui.py
