import time
import threading
import pyRAPL
import pynvml
import psutil  # Für CPU-RAM-Messung


class EnergyMeter:
    def __init__(self, cpu_segment_interval=300, gpu_sampling_interval=0.001, mem_sampling_interval=1.0):
        """
        :param cpu_segment_interval: Interval in Sekunden, nach dem die CPU-Messung "resettiert" wird.
        :param gpu_sampling_interval: Sampling-Intervall für die GPU-Leistungsmessung.
        :param mem_sampling_interval: Sampling-Intervall für die Speicherauslastungs-Messung (RAM/VRAM).
        """
        self.cpu_segment_interval = cpu_segment_interval
        self.gpu_sampling_interval = gpu_sampling_interval
        self.mem_sampling_interval = mem_sampling_interval

        self.cpu_segment_energies = []
        self.cpu_power_samples = []
        self.gpu_samples = []
        self.cpu_memory_samples = []
        self.gpu_memory_samples = []

        self._cpu_segment_stop_event = threading.Event()
        self._gpu_stop_event = threading.Event()
        self._mem_stop_event = threading.Event()

    def _cpu_segment_monitor(self):
        """
        Dieser Thread beendet regelmäßig den aktuellen CPU-Messabschnitt, ermittelt
        die über den Abschnitt aufgenommene Energie sowie die durchschnittliche Leistung,
        speichert diese und startet einen neuen Abschnitt.
        """
        while not self._cpu_segment_stop_event.wait(self.cpu_segment_interval):
            self.cpu_meter.end()
            current_time = time.perf_counter()
            seg_duration = current_time - self.cpu_segment_start_time
            seg_energy = sum(self.cpu_meter.result.pkg) / 1e6 if self.cpu_meter.result.pkg else 0
            self.cpu_segment_energies.append(seg_energy)
            seg_power = seg_energy / seg_duration if seg_duration > 0 else 0
            self.cpu_power_samples.append(seg_power)
            self.cpu_meter = pyRAPL.Measurement('NER_Evaluation')
            self.cpu_meter.begin()
            self.cpu_segment_start_time = time.perf_counter()

    def _sample_gpu_power(self):
        """
        Dieser Thread erfasst in regelmäßigen Abständen (gpu_sampling_interval) die GPU-Leistung.
        """
        while not self._gpu_stop_event.is_set():
            current_time = time.perf_counter()
            power_mW = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle)
            self.gpu_samples.append((current_time, power_mW))
            time.sleep(self.gpu_sampling_interval)

    def _sample_memory_usage(self):
        """
        Dieser Thread misst in regelmäßigen Abständen (mem_sampling_interval)
        die RAM-Auslastung der CPU und den VRAM der GPU.
        """
        while not self._mem_stop_event.is_set():
            mem = psutil.virtual_memory()
            self.cpu_memory_samples.append(mem.used / (1024 ** 2))
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            self.gpu_memory_samples.append(meminfo.used / (1024 ** 2))
            time.sleep(self.mem_sampling_interval)

    def __enter__(self):
        pyRAPL.setup()
        self.cpu_meter = pyRAPL.Measurement('NER_Evaluation')
        self.cpu_meter.begin()
        self.cpu_segment_start_time = time.perf_counter()

        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        self.start_time = time.perf_counter()

        self.gpu_sampling_thread = threading.Thread(target=self._sample_gpu_power)
        self.gpu_sampling_thread.start()

        self._cpu_segment_thread = threading.Thread(target=self._cpu_segment_monitor)
        self._cpu_segment_thread.start()

        self._mem_sampling_thread = threading.Thread(target=self._sample_memory_usage)
        self._mem_sampling_thread.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._gpu_stop_event.set()
        self.gpu_sampling_thread.join()

        self._cpu_segment_stop_event.set()
        self._cpu_segment_thread.join()

        self._mem_stop_event.set()
        self._mem_sampling_thread.join()

        self.cpu_meter.end()
        current_time = time.perf_counter()
        seg_duration = current_time - self.cpu_segment_start_time
        current_seg_energy = sum(self.cpu_meter.result.pkg) / 1e6 if self.cpu_meter.result.pkg else 0
        self.cpu_segment_energies.append(current_seg_energy)
        seg_power = current_seg_energy / seg_duration if seg_duration > 0 else 0
        self.cpu_power_samples.append(seg_power)

        self.end_time = time.perf_counter()
        self.runtime = self.end_time - self.start_time

        self.cpu_energy = sum(self.cpu_segment_energies)
        self.cpu_power_avg = self.cpu_energy / self.runtime if self.runtime > 0 else 0
        self.cpu_power_max = max(self.cpu_power_samples) if self.cpu_power_samples else 0

        if len(self.gpu_samples) < 2:
            self.gpu_energy = 0
            self.gpu_power_avg = 0
            self.gpu_power_max = 0
        else:
            energy = 0
            gpu_power_values = []
            for i in range(len(self.gpu_samples) - 1):
                t0, p0 = self.gpu_samples[i]
                t1, p1 = self.gpu_samples[i + 1]
                dt = t1 - t0
                energy += ((p0 + p1) / 2000) * dt
                gpu_power_values.append(p0)
            gpu_power_values.append(self.gpu_samples[-1][1])
            self.gpu_energy = energy
            self.gpu_power_avg = energy / self.runtime if self.runtime > 0 else 0
            self.gpu_power_max = max(gpu_power_values) / 1000

        pynvml.nvmlShutdown()

        self.cpu_memory_avg = (
                    sum(self.cpu_memory_samples) / len(self.cpu_memory_samples)) if self.cpu_memory_samples else 0
        self.gpu_memory_avg = (
                    sum(self.gpu_memory_samples) / len(self.gpu_memory_samples)) if self.gpu_memory_samples else 0


if __name__ == "__main__":
    with EnergyMeter(cpu_segment_interval=300, gpu_sampling_interval=0.001, mem_sampling_interval=1.0) as meter:
        time.sleep(600)

    print("Laufzeit: {:.2f} s".format(meter.runtime))
    print("CPU Energie: {:.6f} J".format(meter.cpu_energy))
    print("Durchschn. CPU Leistung: {:.6f} W".format(meter.cpu_power_avg))
    print("Max. CPU Leistung: {:.6f} W".format(meter.cpu_power_max))
    print("GPU Energie: {:.6f} J".format(meter.gpu_energy))
    print("Durchschn. GPU Leistung: {:.6f} W".format(meter.gpu_power_avg))
    print("Max. GPU Leistung: {:.6f} W".format(meter.gpu_power_max))
    print("Durchschn. CPU RAM-Auslastung: {:.2f} MB".format(meter.cpu_memory_avg))
    print("Durchschn. GPU VRAM-Auslastung: {:.2f} MB".format(meter.gpu_memory_avg))
