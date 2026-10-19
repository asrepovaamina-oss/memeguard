#!/usr/bin/env bash
# запуск демо одной командой (для Demo Day):
# 1. поднимаю Gradio-приложение с моделью на этом маке
# 2. открываю его наружу бесплатным туннелем
# 3. вписываю адрес туннеля в сайт и передеплоиваю его на Vercel
#
# туннелей два, по очереди:
# - Cloudflare: без ограничения по времени, но ходит через порт 7844,
#   который бывает закрыт в сетях вузов и офисов
# - localhost.run: через ssh (порт 22), без регистрации. пробовала ещё Pinggy
#   через порт 443, но бесплатная версия показывает браузерам страницу-предупреждение,
#   и сайт не может достучаться до модели
#
# запуск:  bash scripts/start_demo.sh
# остановка: Ctrl+C (приложение и туннель закроются вместе со скриптом)

set -euo pipefail
cd "$(dirname "$0")/.."

# если демо уже запускалось (например, в другом окне терминала), останавливаю
# старый запуск: иначе порт 7860 занят старым приложением со старым адресом,
# и сайт не может достучаться до модели
if pgrep -f "Python app.py" >/dev/null || pgrep -f "nokey@localhost.run" >/dev/null \
    || pgrep -f "cloudflared tunnel" >/dev/null; then
  echo "останавливаю предыдущий запуск демо..."
  pkill -f "Python app.py" || true
  pkill -f "cloudflared tunnel" || true
  pkill -f "nokey@localhost.run" || true
  sleep 3
fi

LOGS=$(mktemp -d)
APP_PID=""
TUN_PID=""
trap 'kill $APP_PID $TUN_PID 2>/dev/null || true' EXIT

# root - внешний адрес приложения. он нужен, когда туннель не сообщает Gradio,
# что снаружи https: иначе Gradio отдаёт браузеру http-ссылки, и тот их блокирует
start_app() {
  GRADIO_ROOT_PATH="${1:-}" .venv/bin/python app.py > "$LOGS/app.log" 2>&1 &
  APP_PID=$!
  until curl -s -o /dev/null http://127.0.0.1:7860/; do
    kill -0 "$APP_PID" 2>/dev/null || { cat "$LOGS/app.log"; exit 1; }
    sleep 2
  done
}

echo "запускаю модель (около минуты)..."
start_app

# ждёт, пока в логе туннеля появится адрес, и проверяет, что модель через него
# реально отвечает: бесплатные туннели иногда выдают адрес, но не подключаются
wait_tunnel() {
  local pattern=$1
  URL=""
  for _ in $(seq 1 30); do
    sleep 1
    URL=$(grep -o "$pattern" "$LOGS/tunnel.log" | head -1 || true)
    [ -n "$URL" ] && break
  done
  [ -z "$URL" ] && return 1
  for _ in $(seq 1 10); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$URL/config")" = "200" ] && return 0
    sleep 3
  done
  return 1
}

URL=""
for attempt in 1 2; do
  echo "открываю туннель Cloudflare (попытка $attempt)..."
  # http2, а не QUIC по умолчанию: некоторые сети блокируют QUIC
  cloudflared tunnel --protocol http2 --url http://127.0.0.1:7860 > "$LOGS/tunnel.log" 2>&1 &
  TUN_PID=$!
  wait_tunnel 'https://[a-z0-9-]*\.trycloudflare\.com' && break
  kill "$TUN_PID" 2>/dev/null || true
  URL=""
done

if [ -z "$URL" ]; then
  echo "Cloudflare недоступен в этой сети, открываю туннель localhost.run..."
  ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 \
      -o ConnectTimeout=15 -R 80:127.0.0.1:7860 nokey@localhost.run > "$LOGS/tunnel.log" 2>&1 &
  TUN_PID=$!
  if wait_tunnel 'https://[a-z0-9.-]*\.lhr\.life'; then
    echo "перезапускаю модель с внешним адресом туннеля..."
    kill "$APP_PID" 2>/dev/null || true
    sleep 2
    start_app "$URL"
  else
    URL=""
  fi
fi

if [ -z "$URL" ]; then
  echo "не удалось открыть ни один туннель. проверьте интернет или раздайте его с телефона"
  echo "запасной вариант: локальный интерфейс http://127.0.0.1:7860 работает без интернета"
  wait "$APP_PID"
  exit 1
fi

sed -i '' "s|^export const BACKEND_URL = .*|export const BACKEND_URL = \"$URL\";|" frontend/config.js
echo "обновляю сайт на Vercel..."
(cd frontend && npx --yes vercel deploy --prod --yes) | tail -2

echo
echo "готово. модель: $URL"
echo "сайт обновлён, можно показывать. Ctrl+C чтобы остановить"
wait "$APP_PID"
