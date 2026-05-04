#!/bin/bash
set -e

echo "=== Neural ODE Distributed Training Setup ==="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CTF_DIR="$(dirname "$SCRIPT_DIR")/ctf4science"
DATA_DIR="$CTF_DIR/data"

echo ""
echo "Script directory: $SCRIPT_DIR"
echo "CTF4Science directory: $CTF_DIR"
echo "Data directory: $DATA_DIR"
echo ""

# Step 1: Install Python dependencies
echo "=== Step 1: Installing Python dependencies ==="
pip install -r "$SCRIPT_DIR/requirements.txt"

# Step 2: Install ctf4science if it exists
if [ -d "$CTF_DIR" ]; then
    echo ""
    echo "=== Step 2: Installing ctf4science ==="
    pip install -e "$CTF_DIR"
else
    echo ""
    echo "WARNING: ctf4science not found at $CTF_DIR"
    echo "You may need to clone it separately."
fi

# Step 3: Download and extract data
echo ""
echo "=== Step 3: Downloading data from Google Drive ==="

# Create data directory if it doesn't exist
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# Google Drive file ID
FILE_ID="1BeVNoxs4Xoz6aRKd5LTF-Y7G20XKZCjG"
OUTPUT_FILE="msfr_data.zip"

# Check if msfr data already exists
if [ -d "$DATA_DIR/msfr" ] && [ "$(ls -A $DATA_DIR/msfr 2>/dev/null)" ]; then
    echo "msfr data directory already exists and is not empty."
    read -p "Do you want to re-download? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping download."
    else
        echo "Downloading data..."
        gdown "$FILE_ID" -O "$OUTPUT_FILE"

        echo "Extracting data..."
        unzip -o "$OUTPUT_FILE"
        rm "$OUTPUT_FILE"
    fi
else
    echo "Downloading data..."
    gdown "$FILE_ID" -O "$OUTPUT_FILE"

    echo "Extracting data..."
    unzip -o "$OUTPUT_FILE"
    rm "$OUTPUT_FILE"
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To run the distributed server:"
echo "  cd $SCRIPT_DIR"
echo "  python distributed_server.py --config tuning_config/config_msfr.yaml"
echo ""
echo "To run a worker:"
echo "  cd $SCRIPT_DIR"
echo "  python distributed_worker.py --server http://SERVER_IP:5050 --run-opt run_opt.py"
