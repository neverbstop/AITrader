import time
import psutil

class PerformanceMonitor:
    """Monitors CPU and memory usage of the current process."""
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = time.time()

    def get_memory_usage(self):
        return self.process.memory_info().rss / (1024 ** 2)  # MB

    def get_cpu_usage(self):
        return self.process.cpu_percent(interval=0.1)

    def get_uptime(self):
        return time.time() - self.start_time

    def report(self):
        return {
            "cpu_percent": self.get_cpu_usage(),
            "memory_mb": self.get_memory_usage(),
            "uptime_sec": self.get_uptime()
        }
