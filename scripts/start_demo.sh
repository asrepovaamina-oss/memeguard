#!/usr/bin/env bash
# запуск демо одной командой (для Demo Day):
# 1. поднимаю Gradio-приложение с моделью на этом маке
# 2. открываю его наружу бесплатным туннелем Cloudflare
# 3. вписываю адрес туннеля в сайт и передеплоиваю его на Vercel
#
# запуск:  bash scripts/start_demo.sh
# остановка: Ctrl+C (приложение и туннель закроются вместе со скриптом)

set -euo pipefail
cd "$(dirname "$0")/.."

LOGS=$(mktemp -d)
APP_PID=""
TUN_PID=""
trap 'kill $APP_PID $TUN_PID 2>/dev/null || true' EXIT

echo "запускаю модель (около минуты)..."
.venv/bin/python app.py > "$LOGS/app.log" 2>&1 &
APP_PID=$!
until curl -s -o /dev/null http://127.0.0.1:7860/; do
  kill -0 "$APP_PID" 2>/dev/null || { cat "$LOGS/app.log"; exit 1; }
  sleep 2
done

echo "открываю туннель..."
cloudflared tunnel --url http://127.0.0.1:7860 > "$LOGS/tunnel.log" 2>&1 &
TUN_PID=$!
URL=""
until [ -n "$URL" ]; do
  sleep 1
  URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$LOGS/tunnel.log" | head -1 || true)
done

sed -i '' "s|^export const BACKEND_URL = .*|export const BACKEND_URL = \"$URL\";|" frontend/config.js
echo "обновляю сайт на Vercel..."
(cd frontend && npx --yes vercel deploy --prod --yes) | tail -2

echo
echo "готово. модель: $URL"
echo "сайт обновлён, можно показывать. Ctrl+C чтобы остановить"
wait "$APP_PID"
