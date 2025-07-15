#!/bin/bash

# Regenerate all data files with current documents

echo "Setting up environment..."
export OPENAI_API_KEY=$(cat ~/.openai_key 2>/dev/null || echo "")

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY not found in ~/.openai_key"
    echo "Please create ~/.openai_key file with your OpenAI API key"
    exit 1
fi

echo "Step 1: Regenerating chunks and embeddings from current PDFs..."
python ingest.py docs/ --out-dir data/

echo "Step 2: Starting MCP server in background..."
python doc-mcp.py > mcp_server.log 2>&1 &
MCP_PID=$!
sleep 5  # Give server time to start

echo "Step 3: Starting ngrok tunnel..."
ngrok http 8000 --log-level=info --log=stdout > ngrok.log 2>&1 &
NGROK_PID=$!
sleep 5  # Give ngrok time to start

# Extract ngrok URL from the log
NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | python -c "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])" 2>/dev/null)
if [ -z "$NGROK_URL" ]; then
    echo "Error: Could not get ngrok URL. Is ngrok installed and running?"
    kill $MCP_PID
    exit 1
fi

echo "Ngrok URL: $NGROK_URL"
export HOA_MCP_URL="${NGROK_URL}/mcp"

echo "Step 4: Regenerating draft using deep research..."
python research-with-mcp.py --wait --out draft_new.md

echo "Step 5: Regenerating flags with new draft..."
python ingest.py docs/ --draft draft_new.md --out-dir data/

echo "Stopping services..."
kill $NGROK_PID 2>/dev/null
kill $MCP_PID 2>/dev/null

echo "Done! New draft saved as draft_new.md"
echo "Review draft_new.md and if correct, run: mv draft_new.md draft.md" 