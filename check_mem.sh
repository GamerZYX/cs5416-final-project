#!/bin/bash

# --- Configuration ---

# 1. Define Environment Variables (Assignments)
# Note: These must be in the format KEY=VALUE
ENV_VARS=(
    TOTAL_NODES=3 
    NODE_NUMBER=0 
    NODE_0_IP=132.236.91.188:8000 
    NODE_1_IP=132.236.91.188:8001 
    NODE_2_IP=132.236.91.188:8002 
)

# 2. Define the Actual Program and Arguments
PROGRAM_AND_ARGS=(
    python3
    pipeline.py
)

# 3. Construct the final command array using 'env'
# This ensures a shell-agnostic execution of the environment variables.
PROGRAM_CMD_ARR=(env "${ENV_VARS[@]}" "${PROGRAM_AND_ARGS[@]}")

# The name of the process/command to monitor (the executable program).
PROCESS_NAME="python"

# Sampling interval in seconds
INTERVAL=1
# Output file
LOGFILE="memory_log_$(date +%Y%m%d_%H%M%S).csv"

# --- Functions ---
function cleanup {
    echo -e "\n--- Script terminated. Results saved to $LOGFILE ---"
    # Kill the background process if it's still running
    if [ ! -z "$CHILD_PID" ] && ps -p $CHILD_PID > /dev/null; then
        echo "Attempting to terminate monitored process (PID $CHILD_PID)..."
        kill -SIGTERM $CHILD_PID 2>/dev/null
    fi
    exit 0
}

# Trap signals (like Ctrl+C) to ensure cleanup runs
trap cleanup SIGINT SIGTERM

# --- Main Logic ---
echo "--- Starting Memory Tracker ---"
# Display the command as it will be executed
echo "Command to run: ${PROGRAM_CMD_ARR[@]}" 

# 1. Start the target program in the background
echo "Starting process: $PROCESS_NAME"
# Use the array expansion "${PROGRAM_CMD_ARR[@]}" with the 'env' executable
( 
    # Execute the program, piping its stdout and stderr (using '2>&1') to 'tee'
    # 'tee' will write to the log file AND to the parent script's stdout
    "${PROGRAM_CMD_ARR[@]}" 2>&1 | tee "events.log"
) &
CHILD_PID=$!
echo "Monitored Process PID: $CHILD_PID"

# 2. Wait a moment for the process to fully start and settle
sleep 2

# 3. Create the CSV header
echo "Timestamp,Time_Relative_s,RSS_KB,VSZ_KB" > "$LOGFILE"
START_TIME=$(date +%s.%N)
echo "Tracking memory every ${INTERVAL} second(s). Press Ctrl+C to stop."

# 4. Main tracking loop
while true; do
    # Check if the process is still running
    if ! ps -p $CHILD_PID > /dev/null; then
        echo -e "\nMonitored process (PID $CHILD_PID) has terminated."
        break
    fi

    # Use 'ps' to get RSS and VSZ in KB
    MEMORY_DATA=$(ps -p $CHILD_PID -o rss,vsz | tail -n 1 | awk '{print $1 "," $2}')
    CURRENT_TIME=$(date +%s.%N)
    # Use 'printf' for floating point arithmetic precision instead of 'bc -l'
    TIME_RELATIVE=$(printf "%.3f" $(echo "$CURRENT_TIME - $START_TIME" | bc -l))

    # Log the data
    echo "$CURRENT_TIME,$TIME_RELATIVE,$MEMORY_DATA" >> "$LOGFILE"
    
    # Output status to terminal
    # echo -e "Time: ${TIME_RELATIVE}s | RSS: $(echo "$MEMORY_DATA" | cut -d',' -f1) KB \r"
    
    sleep $INTERVAL
done

# 5. Final cleanup call
cleanup