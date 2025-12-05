import psutil # <-- Added psutil for memory tracking
from typing import Tuple, List
import threading
import os
import time 
import csv

MemoryRecord = Tuple[float, float, str]
MemoryData = List[MemoryRecord]

class MemoryTracker(threading.Thread):
    """
    A thread that continuously samples the memory usage (RSS) 
    of the current process and logs events.
    """
    def __init__(self, interval: float = 0.1, output_file: str = "memory_log.csv"):
        super().__init__()
        self.interval = interval
        self._stop_event = threading.Event()
        self.memory_data: MemoryData = []
        self.process = psutil.Process(os.getpid())
        self.output_file = output_file
        self._current_event_tag = "" # Tag to be logged with the next memory sample

    def run(self):
        """Main sampling loop."""
        print("Memory tracking started...")
        while not self._stop_event.is_set():
            try:
                # Get the Resident Set Size (RSS) in bytes
                rss_bytes = self.process.memory_info().rss
                
                # Convert to MB
                rss_mb = rss_bytes / (1024 * 1024)
                
                # Record the current time, memory usage, and the current event tag
                self.memory_data.append((time.time(), rss_mb, self._current_event_tag))
                
                # Clear the tag after logging it once
                self._current_event_tag = "" 
                
                # Wait for the specified interval
                time.sleep(self.interval)
            except psutil.NoSuchProcess:
                print("Process ended, stopping tracker.")
                break

    def set_event(self, tag: str):
        """Sets a tag to be logged with the next memory sample."""
        self._current_event_tag = tag
        print(f"--- EVENT: {tag} ---")

    def stop(self):
        """Stops the sampling thread and writes data to CSV."""
        self._stop_event.set()
        print("Memory tracking stopped.")

    def write_csv(self):
        """Writes the collected memory data to a CSV file."""
        if not self.memory_data:
            return
            
        start_time = self.memory_data[0][0]
        
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Time_Relative_s", "Memory_MB", "Event_Tag"])
            for timestamp, usage_mb, tag in self.memory_data:
                relative_time = timestamp - start_time
                writer.writerow([f"{relative_time:.3f}", f"{usage_mb:.2f}", tag])
        print(f"Memory log written to {self.output_file}")