import time
import threading
import pyRAPL
import pynvml
import psutil  # Für CPU-RAM-Messung


class EnergyMeter:
    def __init__(self, cpu_segment_interval=300, gpu_sampling_interval=0.001, mem_sampling_interval=1.0):
        """
        :param cpu_segment_interval: Zeitintervall (s), nach dem ein CPU-Messabschnitt beendet wird.
        :param gpu_sampling_interval: Abtastrate der GPU-Leistungsmessung (s).
        :param mem_sampling_interval: Abtastrate der Speicher-Messung (s).
        """
        self.cpu_segment_interval = cpu_segment_interval
        self.gpu_sampling_interval = gpu_sampling_interval
        self.mem_sampling_interval = mem_sampling_interval

        # CPU-Metriken
        self.cpu_segment_energies = []
        self.cpu_power_samples = []

        # GPU-Metriken
        self.gpu_samples = []

        # Speicher-Samples
        self.cpu_memory_samples = []
        self.gpu_memory_samples = []

        # Stop-Events
        self._cpu_segment_stop_event = threading.Event()
        self._gpu_stop_event = threading.Event()
        self._mem_stop_event = threading.Event()

    # --------------------- Hintergrund-Threads ---------------------

    def _cpu_segment_monitor(self):
        """Beendet periodisch den aktuellen CPU-Messabschnitt und startet einen neuen."""
        while not self._cpu_segment_stop_event.wait(self.cpu_segment_interval):
            self.cpu_meter.end()
            current_time = time.perf_counter()
            seg_duration = current_time - self.cpu_segment_start_time
            seg_energy = sum(self.cpu_meter.result.pkg) / 1e6 if self.cpu_meter.result.pkg else 0
            self.cpu_segment_energies.append(seg_energy)
            self.cpu_power_samples.append(seg_energy / seg_duration if seg_duration else 0)

            # Neuen Abschnitt starten
            self.cpu_meter = pyRAPL.Measurement('NER_Evaluation')
            self.cpu_meter.begin()
            self.cpu_segment_start_time = time.perf_counter()

    def _sample_gpu_power(self):
        """Erfasst regelmäßig die GPU-Leistungsaufnahme."""
        while not self._gpu_stop_event.is_set():
            current_time = time.perf_counter()
            power_mW = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle)
            self.gpu_samples.append((current_time, power_mW))
            time.sleep(self.gpu_sampling_interval)

    def _sample_memory_usage(self):
        """Erfasst regelmäßig die RAM- und VRAM-Auslastung."""
        while not self._mem_stop_event.is_set():
            mem = psutil.virtual_memory()
            self.cpu_memory_samples.append(mem.used / (1024 ** 2)) 
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            self.gpu_memory_samples.append(meminfo.used / (1024 ** 2))
            time.sleep(self.mem_sampling_interval)

    # --------------------- Context-Manager ---------------------

    def __enter__(self):
        # CPU-Messung initialisieren
        pyRAPL.setup()
        self.cpu_meter = pyRAPL.Measurement('NER_Evaluation')
        self.cpu_meter.begin()
        self.cpu_segment_start_time = time.perf_counter()

        # GPU-API initialisieren
        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        self.start_time = time.perf_counter()

        # Threads starten
        self.gpu_sampling_thread = threading.Thread(target=self._sample_gpu_power)
        self.gpu_sampling_thread.start()

        self._cpu_segment_thread = threading.Thread(target=self._cpu_segment_monitor)
        self._cpu_segment_thread.start()

        self._mem_sampling_thread = threading.Thread(target=self._sample_memory_usage)
        self._mem_sampling_thread.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Threads sauber stoppen
        self._gpu_stop_event.set()
        self.gpu_sampling_thread.join()

        self._cpu_segment_stop_event.set()
        self._cpu_segment_thread.join()

        self._mem_stop_event.set()
        self._mem_sampling_thread.join()

        # Letzten CPU-Abschnitt abschließen
        self.cpu_meter.end()
        current_time = time.perf_counter()
        seg_duration = current_time - self.cpu_segment_start_time
        current_seg_energy = sum(self.cpu_meter.result.pkg) / 1e6 if self.cpu_meter.result.pkg else 0
        self.cpu_segment_energies.append(current_seg_energy)
        self.cpu_power_samples.append(current_seg_energy / seg_duration if seg_duration else 0)

        # Gesamtlaufzeit
        self.end_time = time.perf_counter()
        self.runtime = self.end_time - self.start_time

        # --------------------- CPU-Kennzahlen ---------------------
        self.cpu_energy = sum(self.cpu_segment_energies)
        self.cpu_power_avg = self.cpu_energy / self.runtime if self.runtime else 0
        self.cpu_power_max = max(self.cpu_power_samples) if self.cpu_power_samples else 0

        # --------------------- GPU-Kennzahlen ---------------------
        if len(self.gpu_samples) < 2:
            self.gpu_energy = self.gpu_power_avg = self.gpu_power_max = 0
        else:
            energy = 0
            gpu_power_values = []
            for (t0, p0), (t1, p1) in zip(self.gpu_samples[:-1], self.gpu_samples[1:]):
                dt = t1 - t0
                energy += ((p0 + p1) / 2000) * dt 
                gpu_power_values.append(p0)
            gpu_power_values.append(self.gpu_samples[-1][1])
            self.gpu_energy = energy
            self.gpu_power_avg = energy / self.runtime if self.runtime else 0
            self.gpu_power_max = max(gpu_power_values) / 1000  # mW → W

        pynvml.nvmlShutdown()

        # --------------------- Speicher-Kennzahlen ---------------------
        self.cpu_memory_avg = (
            sum(self.cpu_memory_samples) / len(self.cpu_memory_samples)
        ) if self.cpu_memory_samples else 0
        self.gpu_memory_avg = (
            sum(self.gpu_memory_samples) / len(self.gpu_memory_samples)
        ) if self.gpu_memory_samples else 0

        self.cpu_memory_max = max(self.cpu_memory_samples) if self.cpu_memory_samples else 0
        self.gpu_memory_max = max(self.gpu_memory_samples) if self.gpu_memory_samples else 0


# --------------------- Beispielnutzung ---------------------

if __name__ == "__main__":
    with EnergyMeter(cpu_segment_interval=300, gpu_sampling_interval=0.001, mem_sampling_interval=1.0) as meter:
        # Hier kommt deine eigentliche Arbeitslast
        time.sleep(600)  # Dummy-Workload

    print(f"Laufzeit: {meter.runtime:.2f} s")
    print(f"CPU-Energie: {meter.cpu_energy:.6f} J")
    print(f"⌀ CPU-Leistung: {meter.cpu_power_avg:.6f} W")
    print(f"Max. CPU-Leistung: {meter.cpu_power_max:.6f} W")
    print(f"GPU-Energie: {meter.gpu_energy:.6f} J")
    print(f"⌀ GPU-Leistung: {meter.gpu_power_avg:.6f} W")
    print(f"Max. GPU-Leistung: {meter.gpu_power_max:.6f} W")
    print(f"⌀ CPU-RAM-Auslastung: {meter.cpu_memory_avg:.2f} MB")
    print(f"Max. CPU-RAM-Auslastung: {meter.cpu_memory_max:.2f} MB")  
    print(f"⌀ GPU-VRAM-Auslastung: {meter.gpu_memory_avg:.2f} MB")
    print(f"Max. GPU-VRAM-Auslastung: {meter.gpu_memory_max:.2f} MB")
