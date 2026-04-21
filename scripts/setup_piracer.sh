#!/usr/bin/env bash

# Stop on error
set -e

echo "🎬 Starting setup..."

echo "🔄 Updating donkeycar repo..."
cd ~/donkeycar
git pull

# echo "Running setup_uv.sh..."
# ./setup_uv.sh

# echo "Activating virtual environment..."
# source .venv/bin/activate

echo "🧹 Cleaning up..."
rm -rf ~/piracerpro

echo "🚗 Creating a new car..."
donkey createcar --path ~/piracerpro
cd ~/piracerpro
cp ~/donkeycar/myconfig.py ~/piracerpro/

echo "🤖 Fetching self-driving model from server..."
HOST=$(hostname)
echo "Hostname: $HOST"

SUFFIX=${HOST#piracerpro-}
MODEL_NAME="dietikon_${SUFFIX}.pt"
echo "Model: $MODEL_NAME"

REMOTE_MODEL="gold:/home/formulausi/Phaenomena/donkeycar/macchinina/models/dietikon_v2/${MODEL_NAME}"
LOCAL_MODEL_DIR="$HOME/piracerpro/models"

rsync --progress -zrhe ssh "$REMOTE_MODEL" "$LOCAL_MODEL_DIR/"

echo "✅ Setup completed!"