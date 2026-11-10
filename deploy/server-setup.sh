#!/usr/bin/env bash
# AI 서버(Ubuntu)에서 한 번 필요한 준비. deploy.sh가 ssh로 실행하며, 다시 실행해도 된다.
set -euo pipefail

# RAM 2GB 서버라 이미지 빌드(pip install) 중에 메모리가 모자라지 않도록 스왑 2GB를 둔다.
if [ -z "$(swapon --show)" ]; then
  echo "  스왑 2GB 추가"
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile >/dev/null
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

if ! command -v docker >/dev/null; then
  echo "  Docker 설치"
  sudo apt-get update -qq
  # docker.io만 설치하면 docker build가 지원 종료된 예전 빌더로 돈다.
  sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker.io docker-buildx curl >/dev/null
  sudo systemctl enable --now docker
fi

echo "  준비 완료: $(sudo docker --version)"
