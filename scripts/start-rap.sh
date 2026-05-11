#!/usr/bin/env zsh
# start-rap.sh — Start all RAP infrastructure and the proxy with ngrok
#
# Starts:
#   1. Qdrant vector database (Podman container with persistent volume)
#   2. LM Studio server + loads the embedding model
#   3. The original proxy (with RAP pipeline integrated + ngrok tunnel)
#
# Usage:
#   ./scripts/start-rap.sh              # Start everything (uses saved ngrok or starts new)
#   ./scripts/start-rap.sh --ngrok URL  # Use a specific ngrok URL (e.g. from ngrok dashboard)
#   ./scripts/start-rap.sh --stop       # Stop everything
#   ./scripts/start-rap.sh --status     # Check status of all services

set -euo pipefail

# --- Configuration ---
QDRANT_CONTAINER="rap-qdrant"
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334
QDRANT_VOLUME="rap_qdrant_storage"
QDRANT_IMAGE="docker.io/qdrant/qdrant:latest"

LMS_PORT=1234
EMBEDDING_MODEL="text-embedding-nomic-embed-text-v1.5-embedding"

PROXY_HOST="127.0.0.1"
PROXY_PORT=9000

NGROK_LINK_FILE="$HOME/.deepseek-cursor-proxy/.ngrok_url"

# --- Colors ---
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo "${GREEN}[RAP]${NC} $1"; }
warn()  { echo "${YELLOW}[RAP]${NC} $1"; }
error() { echo "${RED}[RAP]${NC} $1"; }
header() { echo "${BOLD}${CYAN}$1${NC}"; }

# --- Status mode ---
if [[ "${1:-}" == "--status" ]]; then
    header "=== RAP Stack Status ==="
    echo ""

    # Qdrant
    if podman container exists "$QDRANT_CONTAINER" 2>/dev/null && \
       [[ "$(podman inspect -f '{{.State.Running}}' "$QDRANT_CONTAINER" 2>/dev/null)" == "true" ]]; then
        info "Qdrant:    ✅ running on port $QDRANT_PORT"
    else
        warn "Qdrant:    ❌ not running"
    fi

    # LM Studio
    if curl -sf "http://localhost:${LMS_PORT}/v1/models" > /dev/null 2>&1; then
        info "LM Studio: ✅ running on port $LMS_PORT"
        # Check embedding model
        MODEL_STATUS=$(curl -sf "http://localhost:${LMS_PORT}/api/v1/models" | \
            python3 -c "
import sys, json
data = json.load(sys.stdin)
for m in data.get('models', []):
    if m.get('key') == '$EMBEDDING_MODEL' and len(m.get('loaded_instances', [])) > 0:
        print('loaded')
        sys.exit(0)
print('not loaded')
" 2>/dev/null || echo "unknown")
        if [[ "$MODEL_STATUS" == "loaded" ]]; then
            info "  Model:   ✅ $EMBEDDING_MODEL (loaded)"
        else
            warn "  Model:   ⚠️  $EMBEDDING_MODEL ($MODEL_STATUS)"
        fi
    else
        warn "LM Studio: ❌ not running"
    fi

    # Proxy
    if curl -sf "http://${PROXY_HOST}:${PROXY_PORT}/healthz" > /dev/null 2>&1; then
        info "Proxy:     ✅ running on port $PROXY_PORT"
    elif pgrep -f "deepseek_cursor_proxy" > /dev/null 2>&1; then
        info "Proxy:     ✅ running (original proxy)"
    else
        warn "Proxy:     ❌ not running"
    fi

    # Ngrok
    if [[ -f "$NGROK_LINK_FILE" ]]; then
        SAVED_URL=$(cat "$NGROK_LINK_FILE")
        info "Ngrok URL: $SAVED_URL"
        info "  Cursor Override Base URL: ${SAVED_URL}/v1"
    else
        warn "Ngrok URL: not saved (will be auto-detected on start)"
    fi

    echo ""
    exit 0
fi

# --- Stop mode ---
if [[ "${1:-}" == "--stop" ]]; then
    info "Stopping RAP infrastructure..."

    # Stop proxy
    if pgrep -f "deepseek_cursor_proxy" > /dev/null 2>&1; then
        pkill -f "deepseek_cursor_proxy" && info "Proxy stopped." || true
    fi

    # Unload embedding model from LM Studio
    if command -v lms &> /dev/null && curl -sf "http://localhost:${LMS_PORT}/v1/models" > /dev/null 2>&1; then
        lms unload --all 2>/dev/null && info "LM Studio models unloaded." || true
    fi

    # Stop Qdrant container
    if podman container exists "$QDRANT_CONTAINER" 2>/dev/null; then
        podman stop "$QDRANT_CONTAINER" 2>/dev/null && info "Qdrant stopped." || true
    fi

    info "All services stopped."
    exit 0
fi

# --- Parse ngrok URL argument ---
NGROK_URL_ARG=""
if [[ "${1:-}" == "--ngrok" ]]; then
    if [[ -z "${2:-}" ]]; then
        error "Usage: ./scripts/start-rap.sh --ngrok <URL>"
        exit 1
    fi
    NGROK_URL_ARG="$2"
    # Save it
    mkdir -p "$(dirname "$NGROK_LINK_FILE")"
    echo "$NGROK_URL_ARG" > "$NGROK_LINK_FILE"
    info "Saved ngrok URL: $NGROK_URL_ARG"
    shift 2
fi

header "=== Starting RAP Stack ==="
echo ""

# --- Start Qdrant ---
info "Starting Qdrant vector database..."

if ! podman volume exists "$QDRANT_VOLUME" 2>/dev/null; then
    podman volume create "$QDRANT_VOLUME" > /dev/null
    info "Created volume: $QDRANT_VOLUME"
fi

if podman container exists "$QDRANT_CONTAINER" 2>/dev/null; then
    if [[ "$(podman inspect -f '{{.State.Running}}' "$QDRANT_CONTAINER" 2>/dev/null)" == "true" ]]; then
        info "Qdrant already running."
    else
        podman start "$QDRANT_CONTAINER" > /dev/null
        info "Qdrant container restarted."
    fi
else
    podman pull "$QDRANT_IMAGE" 2>/dev/null || true
    podman run -d \
        --name "$QDRANT_CONTAINER" \
        -p "${QDRANT_PORT}:6333" \
        -p "${QDRANT_GRPC_PORT}:6334" \
        -v "${QDRANT_VOLUME}:/qdrant/storage:z" \
        "$QDRANT_IMAGE" > /dev/null
    info "Qdrant started on port $QDRANT_PORT."
fi

# Wait for Qdrant
for i in {1..30}; do
    if curl -sf "http://localhost:${QDRANT_PORT}/collections" > /dev/null 2>&1; then
        info "Qdrant is ready."
        break
    fi
    [[ $i -eq 30 ]] && { error "Qdrant failed to start."; exit 1; }
    sleep 1
done

# --- Start LM Studio ---
info "Starting LM Studio server..."

if curl -sf "http://localhost:${LMS_PORT}/v1/models" > /dev/null 2>&1; then
    info "LM Studio server already running."
else
    if command -v lms &> /dev/null; then
        lms server start --port "$LMS_PORT" 2>/dev/null || true
        for i in {1..15}; do
            if curl -sf "http://localhost:${LMS_PORT}/v1/models" > /dev/null 2>&1; then
                info "LM Studio server started."
                break
            fi
            [[ $i -eq 15 ]] && { error "LM Studio failed to start."; exit 1; }
            sleep 1
        done
    else
        error "lms CLI not found. Install LM Studio and run it once."
        exit 1
    fi
fi

# Load embedding model
MODEL_LOADED=$(curl -sf "http://localhost:${LMS_PORT}/api/v1/models" | \
    python3 -c "
import sys, json
data = json.load(sys.stdin)
for m in data.get('models', []):
    if m.get('key') == '$EMBEDDING_MODEL' and len(m.get('loaded_instances', [])) > 0:
        print('yes')
        sys.exit(0)
print('no')
" 2>/dev/null || echo "no")

if [[ "$MODEL_LOADED" == "yes" ]]; then
    info "Embedding model already loaded."
else
    info "Loading embedding model (may take time from external storage)..."
    lms load "$EMBEDDING_MODEL" --gpu max 2>/dev/null || {
        warn "Could not auto-load. Model may JIT-load on first request."
    }
    for i in {1..120}; do
        MODEL_LOADED=$(curl -sf "http://localhost:${LMS_PORT}/api/v1/models" | \
            python3 -c "
import sys, json
data = json.load(sys.stdin)
for m in data.get('models', []):
    if m.get('key') == '$EMBEDDING_MODEL' and len(m.get('loaded_instances', [])) > 0:
        print('yes')
        sys.exit(0)
print('no')
" 2>/dev/null || echo "no")
        [[ "$MODEL_LOADED" == "yes" ]] && { info "Embedding model loaded."; break; }
        [[ $i -eq 120 ]] && { warn "Model still loading. Proxy will start in degraded mode."; break; }
        sleep 1
    done
fi

# --- Start Proxy ---
info "Starting proxy with RAP pipeline..."

cd "$(dirname "$0")/.."

# Determine ngrok behavior
NGROK_FLAG="--ngrok"
if [[ -n "$NGROK_URL_ARG" ]]; then
    # User provided a URL — don't start ngrok (they're managing it externally)
    NGROK_FLAG="--no-ngrok"
fi

# Start the original proxy (which now has RAP pipeline integrated)
python -m deepseek_cursor_proxy --port "$PROXY_PORT" $NGROK_FLAG &
PROXY_PID=$!

# Wait for proxy to be ready
for i in {1..15}; do
    if curl -sf "http://${PROXY_HOST}:${PROXY_PORT}/healthz" > /dev/null 2>&1; then
        break
    fi
    sleep 1
done

# Detect ngrok URL
if [[ -n "$NGROK_URL_ARG" ]]; then
    NGROK_URL="$NGROK_URL_ARG"
elif [[ "$NGROK_FLAG" == "--ngrok" ]]; then
    # Wait for ngrok to report its URL
    sleep 3
    NGROK_URL=$(curl -sf "http://127.0.0.1:4040/api/endpoints" 2>/dev/null | \
        python3 -c "
import sys, json
data = json.load(sys.stdin)
for ep in data.get('endpoints', data.get('tunnels', [])):
    url = ep.get('url', ep.get('public_url', ''))
    if url.startswith('https://'):
        print(url)
        sys.exit(0)
    elif url.startswith('http://'):
        print(url)
        sys.exit(0)
print('')
" 2>/dev/null || echo "")
    if [[ -n "$NGROK_URL" ]]; then
        mkdir -p "$(dirname "$NGROK_LINK_FILE")"
        echo "$NGROK_URL" > "$NGROK_LINK_FILE"
    fi
else
    NGROK_URL=""
fi

# --- Print Summary ---
echo ""
header "=== RAP Stack Running ==="
echo ""
info "  Qdrant:     http://localhost:${QDRANT_PORT}"
info "  LM Studio:  http://localhost:${LMS_PORT}"
info "  Proxy:      http://${PROXY_HOST}:${PROXY_PORT}"

if [[ -n "$NGROK_URL" ]]; then
    echo ""
    header "  ┌─────────────────────────────────────────────────────┐"
    header "  │  Cursor 'Override OpenAI Base URL':                  │"
    header "  │  ${NGROK_URL}/v1"
    header "  └─────────────────────────────────────────────────────┘"
    echo ""
    info "  Ngrok URL saved to: $NGROK_LINK_FILE"
else
    echo ""
    info "  Local only (no ngrok). Use --ngrok to provide a URL."
    info "  Or enable ngrok in config.yaml (ngrok: true)"
fi

echo ""
info "  Stop all:   ./scripts/start-rap.sh --stop"
info "  Status:     ./scripts/start-rap.sh --status"
echo ""

# Keep running
wait $PROXY_PID
