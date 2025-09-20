import time
import threading
import pandas as pd
import os
from codecarbon import EmissionsTracker
import pynvml

class EnergyMonitor:
    """
    A context manager to monitor energy consumption of a code block.

    Uses CodeCarbon for overall CPU/GPU energy and pynvml for high-frequency
    GPU power sampling.
    """
    def __init__(self, output_dir="codecarbon_logs"):
        self.output_dir = output_dir
        self.gpu_power_samples = []
        self._tracking_thread = None
        self._stop_event = threading.Event()
        self.tracker = None
        self.results = {}

    def _gpu_power_sampler(self, handle, interval=1):
        """Samples GPU power draw at a given interval."""
        while not self._stop_event.is_set():
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
                self.gpu_power_samples.append(power)
            except pynvml.NVMLError as e:
                print(f"NVML Error: {e}")
                break
            time.sleep(interval)

    def __enter__(self):
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize CodeCarbon tracker
        self.tracker = EmissionsTracker(output_dir=self.output_dir, log_level='warning')
        self.tracker.start()

        # Initialize NVML and start GPU power sampling thread
        try:
            pynvml.nvmlInit()
            # Assuming a single GPU device
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._stop_event.clear()
            self._tracking_thread = threading.Thread(
                target=self._gpu_power_sampler, args=(handle,)
            )
            self._tracking_thread.start()
            print("Started energy monitoring.")
        except pynvml.NVMLError as e:
            print(f"Could not initialize NVML for GPU power sampling: {e}")
            self._tracking_thread = None

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Stop CodeCarbon tracker
        emissions_kg_co2eq = self.tracker.stop()

        # Stop GPU power sampling thread
        if self._tracking_thread:
            self._stop_event.set()
            self._tracking_thread.join()
            pynvml.nvmlShutdown()

        # Process results
        if isinstance(emissions_kg_co2eq, float):
            # In simple/fallback mode, codecarbon returns total emissions in kg CO2eq.
            # We can't get a detailed energy breakdown.
            self.results['emissions_kg_co2eq'] = emissions_kg_co2eq
            self.results['energy_Wh'] = "N/A"
            self.results['energy_CPU_Wh'] = "N/A"
            self.results['energy_GPU_Wh'] = "N/A"

        # The duration is not available in the float return, so we calculate it manually
        self.results['wall_clock_s'] = self.tracker._last_measured_time - self.tracker._start_time

        if self.gpu_power_samples:
            avg_power = sum(self.gpu_power_samples) / len(self.gpu_power_samples)
            self.results['avg_gpu_power_W'] = avg_power
        else:
            self.results['avg_gpu_power_W'] = None

        print("Stopped energy monitoring.")
        # The user might want to handle exceptions, so we don't suppress them
        return False

if __name__ == '__main__':
    # --- Example Usage ---
    print("Running a dummy workload for 10 seconds to demonstrate energy monitoring...")

    monitor = EnergyMonitor()
    with monitor:
        # Simulate a workload
        start_time = time.time()
        while time.time() - start_time < 10:
            # Simulate some computation
            _ = [i*i for i in range(10000)]
            time.sleep(0.1)

    print("\n--- Monitoring Results ---")
    if monitor.results:
        for key, value in monitor.results.items():
            print(f"{key}: {value}")
    else:
        print("No results were captured.")

    print("\nExample finished.")
