#!/bin/bash

DIRECTORY=".venv"
CONFIG_FILE="config.json"
EXAMPLE_CONFIG_FILE="example.config.json"

if [ ! -d "$DIRECTORY" ]; then
  echo "$DIRECTORY does not exist."
  echo "Creating $DIRECTORY."
  python3 -m venv $DIRECTORY
  echo "Installing dependencies..."
  export PATH="$DIRECTORY/bin:$PATH"
  pip3 install -r requirements.txt --no-cache-dir
fi

if [ ! -f "$CONFIG_FILE" ]; then
  echo "$CONFIG_FILE doesn't exist"
  echo "Cloning it from $EXAMPLE_CONFIG_FILE"
  cp $EXAMPLE_CONFIG_FILE $CONFIG_FILE
fi

export PATH="$DIRECTORY/bin:$PATH"
python3 main.py