#!/usr/bin/env bash
# AI 서버(임베딩 서버)를 EC2에 배포한다. 다시 실행하면 최신 코드로 갈아 끼운다.
# usage: deploy/deploy.sh <서버 주소> [SSH 키 파일]
#   예) deploy/deploy.sh 3.35.1.2 ~/Downloads/danditon-ai.pem
# 소스(app/, Dockerfile, requirements.txt)만 올리고 서버에서 이미지를 빌드하므로 GitHub 권한은 필요 없다.
set -euo pipefail

HOST="${1:?usage: deploy/deploy.sh <서버 주소> [SSH 키 파일]}"
KEY="${2:-}"
SSH_USER="${SSH_USER:-ubuntu}"
NAME="danditon-ai"
# 서버 RAM 2GB 중 OS가 쓸 몫을 남긴다. 실사용은 약 1.1GB다.
MEMORY_LIMIT="${MEMORY_LIMIT:-1600m}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)
if [ -n "$KEY" ]; then
  # AWS 키 파일은 권한이 넓으면 ssh가 거부한다.
  chmod 400 "$KEY"
  SSH_OPTS+=(-i "$KEY")
fi
remote() { ssh "${SSH_OPTS[@]}" "$SSH_USER@$HOST" "$@"; }

echo "1/5 서버 준비 (Docker·스왑)"
remote 'bash -s' < "$ROOT/deploy/server-setup.sh"

echo "2/5 소스 올리기"
tar -C "$ROOT" -czf - Dockerfile .dockerignore requirements.txt app \
  | remote "rm -rf ~/$NAME && mkdir -p ~/$NAME && tar -xzf - -C ~/$NAME"

echo "3/5 이미지 빌드 (처음에는 가중치를 받느라 5~10분 걸림)"
remote "cd ~/$NAME && sudo docker build -t $NAME:latest ."

echo "4/5 컨테이너 실행"
remote "sudo docker rm -f $NAME >/dev/null 2>&1 || true
  sudo docker run -d --name $NAME --restart unless-stopped -p 8000:8000 --memory $MEMORY_LIMIT $NAME:latest >/dev/null"

echo "5/5 확인"
# 모델 로딩이 끝날 때까지 /health는 503이다.
if ! remote 'for i in $(seq 1 60); do
    [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8000/health)" = 200 ] && exit 0
    sleep 3
  done; exit 1'; then
  echo "❌ 3분이 지나도 /health가 200이 아니에요. 최근 로그:"
  remote "sudo docker logs --tail 50 $NAME"
  exit 1
fi
remote "curl -s localhost:8000/health"; echo

# 서버에는 Pillow가 없으므로 표준 라이브러리로 작은 PNG를 만들어 /embed를 호출해 본다.
remote 'python3 - > /tmp/danditon-smoke.png <<"PY"
import struct, sys, zlib
w = h = 64
rows = b"".join(b"\x00" + bytes(c for x in range(w) for c in (x * 4, y * 4, 128)) for y in range(h))
def chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
sys.stdout.buffer.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                        + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
PY
curl -s -F image=@/tmp/danditon-smoke.png localhost:8000/embed \
  | python3 -c "import json, sys; d = json.load(sys.stdin); print(\"  /embed 응답: model\", d[\"model\"], \"dim\", d[\"dim\"], \"길이\", len(d[\"embedding\"]))"'

PRIVATE_IP="$(remote "hostname -I | awk '{print \$1}'")"
echo
echo "✅ 배포 완료. BE 환경변수에 넣을 값:"
echo "   EMBED_HOST=http://$PRIVATE_IP:8000"
