#!/usr/bin/env bash
# Set up or update the AI Finance Tutor on an Ubuntu 24.04 EC2 instance (story 11).
#
#   curl -fsSL https://raw.githubusercontent.com/simple-sifu/AI_Finance_Assistant/main/deploy/setup-server.sh -o setup-server.sh
#   bash setup-server.sh
#
# After the first run, use the repo's copy so script fixes arrive with each pull:
#   bash ~/AI_Finance_Assistant/deploy/setup-server.sh
#
# Safe to run again: it adds swap and Docker only if missing, pulls the latest
# code, rebuilds the image and replaces the running container. The first run
# creates ~/finance-assistant.env from .env.example and stops so you can fill in
# the keys and APP_PASSWORD. See deploy/RUNBOOK.md.
#
# Everything runs inside main(), which bash reads in full before running, so the
# git pull below can safely update this very file.

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/simple-sifu/AI_Finance_Assistant.git}"
BRANCH="${BRANCH:-main}"
APP_DIR="${APP_DIR:-$HOME/AI_Finance_Assistant}"
ENV_FILE="${ENV_FILE:-$HOME/finance-assistant.env}"
IMAGE="finance-assistant"
CONTAINER="finance-assistant"

log() { printf '\n==> %s\n' "$*"; }

# True when KEY has a non-blank value in the env file (the app strips whitespace and
# dotenv strips quotes, so KEY="" counts as empty).
has_value() { grep -Eq "^$1=[[:space:]]*[\"']?[^[:space:]\"']" "$ENV_FILE"; }

main() {
    # 1. Swap: building and running torch on a 2 GB instance needs headroom.
    if ! sudo swapon --show | grep -q '/swapfile'; then
        log "Adding a 2 GB swap file"
        sudo fallocate -l 2G /swapfile
        sudo chmod 600 /swapfile
        sudo mkswap /swapfile >/dev/null
        sudo swapon /swapfile
        grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
    fi

    # 2. Docker and git.
    if ! command -v docker >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1; then
        log "Installing Docker and git"
        sudo apt-get update -y
        sudo apt-get install -y docker.io docker-buildx git curl
        sudo systemctl enable --now docker
    fi

    # 3. Code: clone the first time, then fast-forward to the latest $BRANCH.
    if [ -d "$APP_DIR/.git" ]; then
        log "Updating the code ($BRANCH)"
        git -C "$APP_DIR" fetch --quiet origin "$BRANCH"
        git -C "$APP_DIR" checkout --quiet "$BRANCH"
        git -C "$APP_DIR" merge --ff-only --quiet "origin/$BRANCH"
    else
        log "Cloning $REPO_URL"
        git clone --quiet --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
    fi
    log "Code at $(git -C "$APP_DIR" log --oneline -1)"

    # 4. Settings: keys and APP_PASSWORD live only in this file on the server.
    if [ ! -f "$ENV_FILE" ]; then
        cp "$APP_DIR/.env.example" "$ENV_FILE"
        chmod 600 "$ENV_FILE"
        log "Created $ENV_FILE. Fill in OPENAI_API_KEY, ALPHA_VANTAGE_API_KEY, TAVILY_API_KEY and APP_PASSWORD:"
        echo "    nano $ENV_FILE"
        echo "Then run this script again."
        exit 0
    fi
    if ! has_value APP_PASSWORD; then
        echo "ERROR: APP_PASSWORD is empty in $ENV_FILE. Without it the public URL is open to anyone"
        echo "(and can spend your OpenAI credits). Set it with: nano $ENV_FILE"
        exit 1
    fi
    if ! has_value OPENAI_API_KEY; then
        echo "WARNING: OPENAI_API_KEY is empty in $ENV_FILE, so questions will show 'OPENAI_API_KEY is not set'."
    fi

    # 5. Image: builds the index and checks it offline (takes several minutes the first time).
    log "Building the image"
    sudo docker build --tag "$IMAGE:latest" "$APP_DIR"

    # 6. Container: replace the old one; restart on crash and after reboot.
    # The env file is mounted, not passed with --env-file: Docker keeps quotes and
    # inline comments as part of the value, the app's dotenv parser strips them.
    # The app user is uid 1000, the same as ubuntu, so it can read the 600 file.
    log "Starting the container"
    sudo docker rm --force "$CONTAINER" >/dev/null 2>&1 || true
    sudo docker run --detach \
        --name "$CONTAINER" \
        --restart unless-stopped \
        --volume "$ENV_FILE:/app/.env:ro" \
        --publish 80:8501 \
        "$IMAGE:latest" >/dev/null
    sudo docker image prune --force >/dev/null
    # Keep the build cache from filling the 20 GB disk over many rebuilds.
    sudo docker builder prune --force --filter until=168h >/dev/null 2>&1 || true

    # 7. Wait until Streamlit answers its health check.
    log "Waiting for the app"
    for _ in $(seq 1 60); do
        if [ "$(curl -fsS --max-time 2 http://localhost/_stcore/health 2>/dev/null)" = "ok" ]; then
            token="$(curl -fsS --max-time 2 -X PUT http://169.254.169.254/latest/api/token \
                -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' 2>/dev/null || true)"
            ip="$(curl -fsS --max-time 2 -H "X-aws-ec2-metadata-token: $token" \
                http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || true)"
            log "App is up: http://${ip:-<your-elastic-ip>}"
            exit 0
        fi
        sleep 2
    done
    echo "The app did not become healthy. Recent logs:"
    sudo docker logs --tail 50 "$CONTAINER"
    exit 1
}

main "$@"
