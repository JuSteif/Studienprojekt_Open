#!/usr/bin/env python3
import csv
import json
import os
import sys
import time

from flair.data import Sentence
from flair.nn import Classifier
from flair.splitter import SciSpacySentenceSplitter

cur_dir = os.path.dirname(__file__)
top_dir = os.path.abspath(os.path.join(cur_dir, '..'))
if top_dir not in sys.path:
    sys.path.insert(0, top_dir)
from EnergyMeter import EnergyMeter
from F1Score import nx, compute_f1_score_contains_bipartite, build_bipartite_graph

"""
json_file = "bc2gm_converted.json"
csv_file = "bc2gm_results.csv"
class_name = "GENE"
# """

"""
json_file = "linnaeus_converted.json"
csv_file = "linnaeus_results.csv"
class_name = "SPECIES"
# """

# """
json_file = "ncbi_disease_converted.json"
csv_file = "ncbi_disease_results.csv"
class_name = "DISEASE"
# """


COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
RESET_COLOR = "\033[0m"

def load_converted_json(json_file):
    """
    Lädt den konvertierten BC2GM-Datensatz aus der JSON-Datei.
    Jeder Eintrag enthält:
      - id
      - text
      - gold_entities: Liste von Dictionaries mit "start", "end", "label"
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def run_hunflair_global(text, tagger, splitter):
    """
    Erzeugt Sätze aus dem vollständigen Text mittels SciSpacySentenceSplitter,
    bestimmt für jeden Satz den globalen Offset (Position im Originaltext),
    führt die NER-Vorhersage mit HunFlair durch und extrahiert die globalen Entitäten.

    Rückgabe:
      Eine Menge von Tupeln (global_start, global_end, label)
    """
    sentences = splitter.split(text)

    global_offset = 0
    for sentence in sentences:
        sentence_text = sentence.to_plain_string()
        found_idx = text.find(sentence_text, global_offset)
        if found_idx == -1:
            found_idx = global_offset
        sentence.global_offset = found_idx
        global_offset = found_idx + len(sentence_text)

    tagger.predict(sentences)

    global_entities = set()
    for sentence in sentences:
        for entity in sentence.get_labels():
            local_start = entity.data_point.start_position
            local_end = entity.data_point.end_position

            global_start = sentence.global_offset + local_start
            global_end = sentence.global_offset + local_end

            label = entity.value
            if label.startswith(("B-", "I-", "E-", "S-")):
                label = label[2:]
            label = label.upper()

            global_entities.add((global_start, global_end, label))

    return global_entities


def highlight_entities_different_colors(text, colored_entities):
    """
    Erzeugt einen Text, in dem die Bereiche, die in colored_entities angegeben sind,
    mithilfe des jeweiligen Farbcodes hervorgehoben werden.
    Erwartet wird eine Liste von Tupeln (start, end, color_code).
    """
    sorted_entities = sorted(colored_entities, key=lambda x: x[0])
    highlighted = ""
    last_index = 0
    for (start, end, color) in sorted_entities:
        highlighted += text[last_index:start]
        highlighted += f"{color}{text[start:end]}{RESET_COLOR}"
        last_index = end
    highlighted += text[last_index:]
    return highlighted


def main():
    num_runs = 10
    results = []

    tagger = Classifier.load("hunflair")
    splitter = SciSpacySentenceSplitter()

    for run in range(num_runs):
        corpus = load_converted_json(json_file)

        with EnergyMeter() as meter:
            for i, entry in enumerate(corpus):
                print(f"Progress: {i / len(corpus) * 100:.2f}%")

                text = entry["text"]
                pred_entities = run_hunflair_global(text, tagger, splitter)
                entry["pred_entities"] = list(pred_entities)

                # colored = highlight_entities_different_colors(text, [(start, end, COLOR_GREEN) for (start, end, label) in pred_entities])
                # print(colored)

        f1, precision, recall = compute_f1_score_contains_bipartite(corpus, class_name)

        runtime = meter.runtime
        cpu_energy = meter.cpu_energy
        gpu_energy = meter.gpu_energy
        cpu_power_avg = meter.cpu_power_avg
        gpu_power_avg = meter.gpu_power_avg
        gpu_power_max = meter.gpu_power_max
        cpu_memory_avg = meter.cpu_memory_avg
        gpu_memory_avg = meter.gpu_memory_avg

        results.append((runtime, f1, precision, recall, cpu_energy, gpu_energy,
                        cpu_power_avg, gpu_power_avg, gpu_power_max, cpu_memory_avg, gpu_memory_avg))
        print(f"Run {run + 1:3d}: Runtime {runtime:.4f} s, F1 {f1:.4f}, Precision {precision:.4f}, Recall {recall:.4f}")

    avg_runtime = sum(r[0] for r in results) / num_runs
    avg_f1 = sum(r[1] for r in results) / num_runs
    avg_precision = sum(r[2] for r in results) / num_runs
    avg_recall = sum(r[3] for r in results) / num_runs
    avg_cpu_energy = sum(r[4] for r in results) / num_runs
    avg_gpu_energy = sum(r[5] for r in results) / num_runs
    avg_cpu_power_avg = sum(r[6] for r in results) / num_runs
    avg_gpu_power_avg = sum(r[7] for r in results) / num_runs
    avg_gpu_power_max = sum(r[8] for r in results) / num_runs
    avg_cpu_memory_avg = sum(r[9] for r in results) / num_runs
    avg_gpu_memory_avg = sum(r[10] for r in results) / num_runs

    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Run", "Runtime (s)", "F1 Score (bipartite contains rule)", "Precision", "Recall",
                         "CPU Energy (J)", "GPU Energy (J)", "CPU Power Avg (W)", "GPU Power Avg (W)",
                         "GPU Max Power (W)", "CPU Memory Avg (MB)", "GPU Memory Avg (MB)"])
        for i, res in enumerate(results):
            writer.writerow([i + 1,
                             f"{res[0]:.4f}",
                             f"{res[1]:.4f}",
                             f"{res[2]:.4f}",
                             f"{res[3]:.4f}",
                             f"{res[4]:.4f}",
                             f"{res[5]:.4f}",
                             f"{res[6]:.4f}",
                             f"{res[7]:.4f}",
                             f"{res[8]:.4f}",
                             f"{res[9]:.2f}",
                             f"{res[10]:.2f}"])
        writer.writerow(["Average",
                         f"{avg_runtime:.4f}",
                         f"{avg_f1:.4f}",
                         f"{avg_precision:.4f}",
                         f"{avg_recall:.4f}",
                         f"{avg_cpu_energy:.4f}",
                         f"{avg_gpu_energy:.4f}",
                         f"{avg_cpu_power_avg:.4f}",
                         f"{avg_gpu_power_avg:.4f}",
                         f"{avg_gpu_power_max:.4f}",
                         f"{avg_cpu_memory_avg:.2f}",
                         f"{avg_gpu_memory_avg:.2f}"])
    print(f"Evaluation results saved to {csv_file}")


if __name__ == "__main__":
    main()
