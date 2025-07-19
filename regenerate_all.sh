#!/bin/bash

# Regenerate all data files with current documents
# Enhanced version with comprehensive logging and error handling

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/regenerate_${TIMESTAMP}.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up processes..."
    
    if [ -n "${NGROK_PID:-}" ]; then
        log_info "Stopping ngrok (PID: $NGROK_PID)..."
        kill "$NGROK_PID" 2>/dev/null || true
    fi
    
    if [ -n "${MCP_PID:-}" ]; then
        log_info "Stopping MCP server (PID: $MCP_PID)..."
        kill "$MCP_PID" 2>/dev/null || true
    fi
    
    # Wait a moment for processes to terminate
    sleep 2
    
    # Force kill if still running
    if [ -n "${NGROK_PID:-}" ] && kill -0 "$NGROK_PID" 2>/dev/null; then
        log_warning "Force killing ngrok..."
        kill -9 "$NGROK_PID" 2>/dev/null || true
    fi
    
    if [ -n "${MCP_PID:-}" ] && kill -0 "$MCP_PID" 2>/dev/null; then
        log_warning "Force killing MCP server..."
        kill -9 "$MCP_PID" 2>/dev/null || true
    fi
}

# Error handling
error_handler() {
    local exit_code=$?
    local line_number=$1
    log_error "Script failed at line $line_number with exit code $exit_code"
    cleanup
    exit "$exit_code"
}

# Set up error handling and cleanup
trap 'error_handler $LINENO' ERR
trap cleanup EXIT

# Create log directory
mkdir -p "$LOG_DIR"

# Start logging
log_info "=" * 60
log_info "HOA Document Regeneration Script Starting"
log_info "Timestamp: $(date)"
log_info "Log file: $LOG_FILE"
log_info "=" * 60

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to wait for a service to be ready
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=${3:-30}
    local attempt=1
    
    log_info "Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" >/dev/null 2>&1; then
            log_success "$service_name is ready!"
            return 0
        fi
        
        log_info "Attempt $attempt/$max_attempts: $service_name not ready yet..."
        sleep 2
        ((attempt++))
    done
    
    log_error "$service_name failed to start after $max_attempts attempts"
    return 1
}

# Function to extract ngrok URL with retry
get_ngrok_url() {
    local max_attempts=10
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        log_info "Attempting to get ngrok URL (attempt $attempt/$max_attempts)..."
        
        # Try to get the URL from ngrok API
        local ngrok_url
        ngrok_url=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | \
                   python3 -c "import sys, json; 
                   try: 
                       tunnels = json.load(sys.stdin)['tunnels']
                       if tunnels: 
                           print(tunnels[0]['public_url'])
                       else: 
                           sys.exit(1)
                   except (KeyError, IndexError, json.JSONDecodeError): 
                       sys.exit(1)" 2>/dev/null)
        
        if [ -n "$ngrok_url" ]; then
            log_success "Ngrok URL obtained: $ngrok_url"
            echo "$ngrok_url"
            return 0
        fi
        
        log_warning "Could not get ngrok URL, retrying in 3 seconds..."
        sleep 3
        ((attempt++))
    done
    
    log_error "Failed to get ngrok URL after $max_attempts attempts"
    return 1
}

# Check prerequisites
log_info "Checking prerequisites..."

if ! command_exists python3; then
    log_error "Python 3 is required but not installed"
    exit 1
fi

if ! command_exists ngrok; then
    log_error "ngrok is required but not installed"
    log_info "Install ngrok from: https://ngrok.com/download"
    exit 1
fi

# Check for required files
if [ ! -d "docs" ]; then
    log_error "docs/ directory not found"
    exit 1
fi

if [ ! -f "ingest.py" ]; then
    log_error "ingest.py not found"
    exit 1
fi

if [ ! -f "doc-mcp.py" ]; then
    log_error "doc-mcp.py not found"
    exit 1
fi

if [ ! -f "research-with-mcp.py" ]; then
    log_error "research-with-mcp.py not found"
    exit 1
fi

log_success "All prerequisites satisfied"

# Step 1: Environment Setup
log_info "Step 1: Setting up environment..."
export OPENAI_API_KEY=$(cat ~/.openai_key 2>/dev/null || echo "")

if [ -z "$OPENAI_API_KEY" ]; then
    log_error "OPENAI_API_KEY not found in ~/.openai_key"
    log_info "Please create ~/.openai_key file with your OpenAI API key"
    exit 1
fi

log_success "OpenAI API key loaded"

# Step 2: Regenerate chunks and embeddings
log_info "Step 2: Regenerating chunks and embeddings from current PDFs..."
if ! python3 ingest.py docs/ --out-dir data/ 2>&1 | tee -a "$LOG_FILE"; then
    log_error "Failed to regenerate chunks and embeddings"
    exit 1
fi
log_success "Chunks and embeddings regenerated"

# Step 3: Start MCP server
log_info "Step 3: Starting MCP server in background..."
python3 doc-mcp.py > "${LOG_DIR}/mcp_server_${TIMESTAMP}.log" 2>&1 &
MCP_PID=$!
log_info "MCP server started with PID: $MCP_PID"

# Wait for MCP server to be ready
if ! wait_for_service "http://localhost:8000" "MCP server" 30; then
    log_error "MCP server failed to start"
    exit 1
fi

# Step 4: Start ngrok tunnel
log_info "Step 4: Starting ngrok tunnel..."
ngrok http 8000 --log-level=info --log=stdout > "${LOG_DIR}/ngrok_${TIMESTAMP}.log" 2>&1 &
NGROK_PID=$!
log_info "Ngrok started with PID: $NGROK_PID"

# Wait for ngrok to be ready
sleep 5

# Get ngrok URL
NGROK_URL=$(get_ngrok_url)
if [ $? -ne 0 ]; then
    log_error "Failed to get ngrok URL"
    exit 1
fi

export HOA_MCP_URL="${NGROK_URL}/mcp"
log_info "MCP URL: $HOA_MCP_URL"

# Step 5: Regenerate draft using deep research
log_info "Step 5: Regenerating draft using deep research..."
log_info "Using enhanced logging for research process..."

# Run research with comprehensive logging
if ! python3 research-with-mcp.py \
    --wait \
    --out "draft_new.json" \
    --log-level "INFO" \
    --log-file "${LOG_DIR}/research_${TIMESTAMP}.log" 2>&1 | tee -a "$LOG_FILE"; then
    log_error "Failed to regenerate draft"
    exit 1
fi

log_success "Draft regeneration completed"

# Step 6: Regenerate flags with new draft using LLM judge
log_info "Step 6: Regenerating flags with new draft (using LLM judge)..."
if ! python3 ingest.py docs/ --draft draft_new.json --out-dir data/ --use-llm-judge 2>&1 | tee -a "$LOG_FILE"; then
    log_error "Failed to regenerate flags"
    exit 1
fi
log_success "Flags regenerated with LLM judge"

# Step 7: Cleanup and completion
log_info "Step 7: Cleaning up services..."
cleanup

# Final status
log_success "=" * 60
log_success "Regeneration completed successfully!"
log_success "New draft saved as: draft_new.json"
log_success "Log files saved in: $LOG_DIR"
log_success "=" * 60

log_info "Next steps:"
log_info "1. Review draft_new.json"
log_info "2. If correct, run: mv draft_new.json draft.json"
log_info "3. Check log files in $LOG_DIR for detailed information"

# Optional: Show log file locations
log_info "Log files created:"
log_info "  - Main log: $LOG_FILE"
log_info "  - MCP server: ${LOG_DIR}/mcp_server_${TIMESTAMP}.log"
log_info "  - Ngrok: ${LOG_DIR}/ngrok_${TIMESTAMP}.log"
log_info "  - Research: ${LOG_DIR}/research_${TIMESTAMP}.log" 