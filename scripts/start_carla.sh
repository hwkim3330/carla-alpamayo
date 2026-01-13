#!/bin/bash
# Start CARLA simulator
# Usage: ./start_carla.sh [CARLA_PATH]

CARLA_PATH="${1:-/opt/carla-simulator}"

if [ ! -d "$CARLA_PATH" ]; then
    echo "CARLA not found at: $CARLA_PATH"
    echo "Usage: ./start_carla.sh [CARLA_PATH]"
    echo ""
    echo "Common CARLA paths:"
    echo "  - /opt/carla-simulator"
    echo "  - ~/CARLA_0.9.15"
    echo "  - ~/carla"
    exit 1
fi

echo "Starting CARLA from: $CARLA_PATH"

# Set display if running headless
if [ -z "$DISPLAY" ]; then
    export DISPLAY=:0
fi

# Start CARLA with optimal settings
cd "$CARLA_PATH"

# Check for low quality mode flag
if [ "$2" == "--low" ]; then
    echo "Starting in low quality mode..."
    ./CarlaUE4.sh -quality-level=Low -RenderOffScreen &
else
    ./CarlaUE4.sh -prefernvidia &
fi

CARLA_PID=$!
echo "CARLA started with PID: $CARLA_PID"
echo "Waiting for CARLA to initialize..."
sleep 10

echo "CARLA should be ready. Connect on localhost:2000"
echo "Press Ctrl+C to stop CARLA"

wait $CARLA_PID
