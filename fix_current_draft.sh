#!/bin/bash

# Fix the current draft by removing references to non-existent documents

echo "Setting up environment..."
export OPENAI_API_KEY=$(cat ~/.openai_key 2>/dev/null || echo "")

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY not found in ~/.openai_key"
    echo "Please create ~/.openai_key file with your OpenAI API key"
    exit 1
fi

echo "Step 1: Backing up current draft..."
cp draft.md draft_backup_$(date +%Y%m%d_%H%M%S).md

echo "Step 2: Checking which documents exist..."
ls -1 docs/*.pdf | sed 's/docs\///' | sed 's/\.pdf//'

echo "Step 3: Removing Financial Policies section (no corresponding PDF)..."
python -c "
import re

# Read the draft
with open('draft.md', 'r') as f:
    content = f.read()

# Find where Part VI starts and remove everything after Part V
pattern = r'## Part VI: Financial Policies.*$'
content_fixed = re.sub(pattern, '', content, flags=re.DOTALL)

# Save the fixed draft
with open('draft.md', 'w') as f:
    f.write(content_fixed.strip())

print('Removed Financial Policies section')
"

echo "Step 4: Regenerating chunks and embeddings..."
python ingest.py docs/ --out-dir data/

echo "Step 5: Regenerating flags with fixed draft..."
python ingest.py docs/ --draft draft.md --out-dir data/

echo "Done! Your draft.md has been fixed to match existing documents."
echo "The Streamlit app should now work correctly." 