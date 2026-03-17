#!/bin/bash
echo "Starting Transcriber..."
python3 start.py
if [ $? -ne 0 ]; then
    echo ""
    echo "[error] Something went wrong. See message above."
    echo "If Python is not installed: https://www.python.org/downloads/"
    read -p "Press Enter to close..."
fi
