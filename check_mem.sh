#!/bin/bash

# --- Configuration ---
# The program you want to run and its arguments
# Example: PROGRAM_CMD="python my_script.py --data large_file.txt"
PROGRAM_CMD="$@"
# The name of the process/command to grep for to find the PID.
# For Python scripts, it might be 'python' or the script name itself.
# For simple commands, use the command name (e.g., 'sleep').
# NOTE: If your command is complex, you might need to adjust the grep pattern.
PROCESS_NAME=$(echo "$@" | awk '{print $1}')
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
echo "Command to run: ${PROGRAM_CMD}"
# 1. Start the target program in the background
echo "Starting process: $PROCESS_NAME"
$PROGRAM_CMD &
CHILD_PID=$!
echo "Monitored Process PID: $CHILD_PID"
# 2. Wait a moment for the process to fully start and settle
sleep 2
# 3. Create the CSV header
echo "Timestamp,Time_Relative_s,RSS_KB,VSZ_KB" > $LOGFILE
START_TIME=$(date +%s.%N)
echo "Tracking memory every ${INTERVAL} second(s). Press Ctrl+C to stop."
# 4. Main tracking loop
while true; do
    # Check if the process is still running
    if ! ps -p $CHILD_PID > /dev/null; then
        echo -e "\nMonitored process (PID $CHILD_PID) has terminated."
        break
    fi
    # Use 'ps' to get RSS (Resident Set Size) and VSZ (Virtual Memory Size) in KB
    # RSS: physical memory (RAM) used
    # VSZ: total virtual memory used
    # 'tail -n 1' skips the header, 'awk' formats the output
    MEMORY_DATA=$(ps -p $CHILD_PID -o rss,vsz | tail -n 1 | awk '{print $1 "," $2}')
    CURRENT_TIME=$(date +%s.%N)
    TIME_RELATIVE=$(echo "$CURRENT_TIME - $START_TIME" | bc -l)
    # Log the data
    echo "$CURRENT_TIME,$TIME_RELATIVE,$MEMORY_DATA" >> $LOGFILE
    # Output status to terminal
    echo -e "Time: ${TIME_RELATIVE}s | RSS: $(echo $MEMORY_DATA | cut -d',' -f1) KB \r"
    sleep $INTERVAL
done
# 5. Final cleanup call
cleanup





