#!/usr/bin/env python3
import os
import sys
import csv
import json
import time
import gc
import dotenv
import torch
import transformers
from flair.splitter import SciSpacySentenceSplitter

cur_dir = os.path.dirname(__file__)
top_dir = os.path.abspath(os.path.join(cur_dir, '..'))
if top_dir not in sys.path:
    sys.path.insert(0, top_dir)

from EnergyMeter import EnergyMeter
from F1Score import compute_f1_score_contains_bipartite

"""
csv_file = "bc2gm_llama_batch_results.csv"
json_file = "bc2gm_converted.json"
system_prompt = {"role": "system", "content": "You are a biomedical NER tool specialized in gene annotation."}
user_prompt_upper = "Extract only the gene names from the following text:"
user_prompt_lower = "Use the format:\nGENE1, GENE2, GENE3, ..."
class_name = "GENE"
# """

"""
csv_file = "linnaeus_llama_batch_results.csv"
json_file = "linnaeus_converted.json"
system_prompt = {"role": "system", "content": "You are a biomedical NER tool specialized in species annotation."}
user_prompt_upper = "Extract only the species names from the following text:"
user_prompt_lower = "Use the format:\nSPECIES1, SPECIES2, SPECIES3, ..."
class_name = "SPECIES"
# """

# """
csv_file = "ncbi_disease_llama_batch_results.csv"
json_file = "ncbi_disease_converted.json"
system_prompt = {"role": "system", "content": "You are a biomedical NER tool specialized in disease annotation."}
user_prompt_upper = "Extract only the disease names from the following text:"
user_prompt_lower = "Use the format:\nDISEASE1, DISEASE2, DISEASE3, ..."
class_name = "DISEASE"
# """

def load_converted_json(json_file):
    """
    Lädt den konvertierten Datensatz.
    Jeder Eintrag hat mindestens:
      - "id"
      - "text"
      - "gold_entities": Liste von Dictionaries mit "start", "end", "label"
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def find_all_occurrences(text, substring):
    """
    Findet alle Indizes, an denen 'substring' in 'text' vorkommt.
    """
    indices = []
    start = 0
    while True:
        index = text.find(substring, start)
        if index == -1:
            break
        indices.append(index)
        start = index + 1
    return indices

def try_move_model(pipe, device):
    """
    Versucht, das Modell des Pipelines auf das angegebene Device zu verschieben.
    Falls ein RuntimeError auftritt, der durch Accelerate Offloading verursacht wird,
    wird eine Warnung ausgegeben und die Bewegung übersprungen.
    """
    try:
        pipe.model.to(device)
    except RuntimeError as re:
        if "You can't move a model that has some modules offloaded" in str(re):
            print(
                "Warning: Modell kann wegen Accelerate Offloading nicht verschoben werden. Überspringe das Modell-Move.")
        else:
            raise re

def run_llama_model(composite_text, pipe):
    """
    Erstellt einen Chat-Prompt, der das LLama-Modell anweist, ausschließlich
    die gewünschten Klassen (z. B. GENE/SPECIES/DISEASE) aus dem übergebenen Composite-Text zu extrahieren.
    Erwartet wird eine kommaseparierte Ausgabe.
    """
    messages = [
        {"role": "system", "content": system_prompt["content"]},
        {"role": "user", "content": (
            f"{user_prompt_upper}\n\n{composite_text}\n\n{user_prompt_lower}"
        )}
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    terminators = [
        pipe.tokenizer.eos_token_id,
        pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    with torch.no_grad():
        outputs = pipe(
            prompt,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.3,
            top_p=0.4,
        )
    generated_text = outputs[0]["generated_text"][len(prompt):].strip()
    return generated_text

def process_sentence_batch(batch, pipe, delimiter="\n===SENT_SPLIT===\n"):
    """
    Nimmt eine Liste von maximal 10 Sätzen (jeder Satz als Dict mit "text") und
    fasst diese zu einem Composite-Text zusammen. Dabei wird für jeden Satz der Start- und Endoffset
    im Composite-Text notiert. Anschließend wird das LLama-Modell global auf den Composite-Text angewendet
    und anhand der Boundaries werden die Vorhersagen den einzelnen Sätzen zugeordnet.
    """
    composite_text = ""
    boundaries = []
    for i, sent in enumerate(batch):
        text = sent["text"]
        if i > 0:
            composite_text += delimiter
        start_pos = len(composite_text)
        composite_text += text
        end_pos = len(composite_text)
        boundaries.append({"index": i, "start": start_pos, "end": end_pos, "text": text})

    try:
        global_output = run_llama_model(composite_text, pipe)
    except Exception as e:
        if "CUDA" in str(e):
            print(f"CUDA-Fehler in Sentence-Batch: {e}. Batch wird als leer bewertet.")
            global_output = ""
        else:
            raise

    batch_predictions = {i: [] for i in range(len(batch))}
    predictions = [pred.strip() for pred in global_output.split(",") if pred.strip()]
    for pred in predictions:
        occurrences = find_all_occurrences(composite_text, pred)
        for occ in occurrences:
            for b in boundaries:
                if b["start"] <= occ < b["end"]:
                    local_start = occ - b["start"]
                    local_end = local_start + len(pred)
                    batch_predictions[b["index"]].append((local_start, local_end, class_name))
                    break

    for i, sent in enumerate(batch):
        sent["pred_entities"] = batch_predictions.get(i, [])
    return batch

def main():
    dotenv.load_dotenv()
    hf_token = os.getenv("HUGGINGFACE")
    if not hf_token:
        sys.exit("Umgebungsvariable HUGGINGFACE (Token) nicht gefunden.")

    model_id = "ContactDoctor/Bio-Medical-Llama-3-8B"
    num_runs = 1
    results = []
    batch_size = 5

    splitter = SciSpacySentenceSplitter()

    batch_count = 0
    for run in range(num_runs):
        pipe = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            token=hf_token
        )
        corpus_run = load_converted_json(json_file)

        with EnergyMeter() as meter:
            for i, sample in enumerate(corpus_run):
                print(f"Processing sample sentence splitter{num_runs}: {i / len(corpus_run) * 100:.2f}%")

                text = sample["text"]
                sentences = splitter.split(text)
                processed_sentences = []
                for sentence in sentences:
                    sent_text = sentence.to_plain_string()
                    start = text.find(sent_text)
                    if start == -1:
                        start = 0
                    end = start + len(sent_text)
                    processed_sentences.append({
                        "text": sent_text,
                        "start": start,
                        "end": end
                    })
                sample["sentences"] = processed_sentences

            for i, sample in enumerate(corpus_run):
                print(f"Processing sample prediction (Run {num_runs}): {i / len(corpus_run) * 100:.2f}%")
                processed_batches = []
                sents = sample.get("sentences", [])
                for j in range(0, len(sents), batch_size):
                    print(f"Processing batch {j}/{len(sents)}")
                    batch = sents[j:j + batch_size]
                    batch = process_sentence_batch(batch, pipe, delimiter="\n===SENT_SPLIT===\n")
                    processed_batches.extend(batch)
                    batch_count += 1
                    torch.cuda.empty_cache()
                    gc.collect()
                    if batch_count % 100 == 0:
                        try_move_model(pipe, "cpu")
                        torch.cuda.empty_cache()
                        gc.collect()
                        try_move_model(pipe, "cuda")
                sample["pred_entities"] = []
                for sent in processed_batches:
                    sample["pred_entities"].extend(sent.get("pred_entities", []))

        runtime = meter.runtime
        cpu_energy = meter.cpu_energy
        gpu_energy = meter.gpu_energy
        cpu_power_avg = getattr(meter, "cpu_power_avg", 0)
        gpu_power_avg = getattr(meter, "gpu_power_avg", 0)
        gpu_power_max = getattr(meter, "gpu_power_max", 0)
        cpu_memory_avg = getattr(meter, "cpu_memory_avg", 0)
        gpu_memory_avg = getattr(meter, "gpu_memory_avg", 0)

        f1, precision, recall = compute_f1_score_contains_bipartite(corpus_run, class_name)
        results.append((runtime, f1, precision, recall,
                        cpu_energy, gpu_energy,
                        cpu_power_avg, gpu_power_avg, gpu_power_max,
                        cpu_memory_avg, gpu_memory_avg))
        print(f"Run {run + 1:2d}: Runtime {runtime:.4f} s, F1 {f1:.4f}, Precision {precision:.4f}, Recall {recall:.4f}")

    avg_runtime = sum(r[0] for r in results) / num_runs
    avg_f1 = sum(r[1] for r in results) / num_runs
    avg_precision = sum(r[2] for r in results) / num_runs
    avg_recall = sum(r[3] for r in results) / num_runs
    avg_cpu_energy = sum(r[4] for r in results) / num_runs
    avg_gpu_energy = sum(r[5] for r in results) / num_runs
    avg_cpu_power_avg = sum(r[6] for r in results) / num_runs
    avg_gpu_power_avg = sum(r[7] for r in results) / num_runs
    avg_gpu_power_max = sum(r[8] for r in results) / num_runs
    avg_cpu_memory = sum(r[9] for r in results) / num_runs
    avg_gpu_memory = sum(r[10] for r in results) / num_runs

    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Run", "Runtime (s)", "F1 Score", "Precision", "Recall",
                         "CPU Energy (J)", "GPU Energy (J)", "CPU Power (avg, W)",
                         "GPU Power (avg, W)", "GPU Power (max, W)",
                         "CPU RAM (avg, MB)", "GPU VRAM (avg, MB)"])
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
                         f"{avg_cpu_memory:.2f}",
                         f"{avg_gpu_memory:.2f}"])
    print(f"Evaluation results saved to {csv_file}")


if __name__ == "__main__":
    main()
