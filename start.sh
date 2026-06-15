#!/bin/bash

# 기존 프로세스 정리
lsof -ti :7788 | xargs kill -9 2>/dev/null
pkill -f cloudflared 2>/dev/null
sleep 1

# Flask 앱 시작
echo "→ 앱 시작 중..."
python3 "$(dirname "$0")/app.py" > /tmp/nuki_app.log 2>&1 &

# rembg 모델 로드 대기
echo "→ rembg 모델 로드 중 (약 10초)..."
sleep 12

# Cloudflare 터널 시작
echo "→ 터널 연결 중..."
cloudflared tunnel --url http://localhost:7788 > /tmp/nuki_tunnel.log 2>&1 &
sleep 8

# URL 출력
URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' /tmp/nuki_tunnel.log | tail -1)
echo ""
echo "✅ 준비 완료!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   $URL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "종료하려면 Ctrl+C"

# 터널 로그 실시간 출력 (연결 유지)
tail -f /tmp/nuki_tunnel.log
