import os
import glob
from pathlib import Path
import json
import pandas as pd
from tqdm.auto import tqdm
import datasets
from mushroom import Mushroom
import torch

# CACHE_DIR = "/import/nlp/jchim/hf-cache/"
CACHE_DIR = "/import/nlp-datasets/LLMs/"
os.environ["HF_HOME"] = CACHE_DIR

mushroom = Mushroom()
out_dir = "/homes/cwj01/projects/mushroom/raw/"
os.makedirs(out_dir, exist_ok=True)

#### Find and process original aci-bench #####
out_path = os.path.join(out_dir, "original.json")
if not os.path.exists(out_path):
    print("Processing original...")
    filename = glob.glob("/homes/cwj01/notes_*.json")[0]  # we only need 1
    encounter_ids = set()
    with open(filename, "r") as f:
        data = json.load(f)
        for k, v in data.items():
            # key - encounter_id and relationship and information type
            # v - str containing note
            encounter_id = k.split("_")[0]
            encounter_ids.add(encounter_id)
    dataset = datasets.load_dataset("ClinicianFOCUS/ACI-Bench-Refined")
    df = pd.concat([pd.DataFrame(dataset[split]) for split in dataset])
    df = df[df.encounter_id.isin(encounter_ids)]
    notes = df.note.tolist()

    with open(out_path, "w") as f:
        json.dump(
            {
                "encounter_ids": df.encounter_id.tolist(),
                "notes": notes,
                "decomposed_raw": mushroom.decompose(
                    notes, batch_size=12, max_new_tokens=2000
                ),
            },
            f,
        )
print("Finished original.")
##########


for filepath in glob.glob("/homes/cwj01/notes_claude*.json") + glob.glob(
    "/homes/cwj01/notes_Mixtral*.json"
):
    filename = Path(filepath).name
    out_path = os.path.join(out_dir, filename)
    if os.path.exists(out_path):
        continue

    print(f"Processing {filename}...")

    with open(filepath, "r") as f:
        data = json.load(f)

    # keys and values are ordered >= py3.6
    notes = list(data.values())
    encounter_ids = list(data.keys())

    decomposed_notes = mushroom.decompose(notes, batch_size=12, max_new_tokens=2000)

    with open(out_path, "w") as f:
        json.dump(
            {
                "encounter_ids": encounter_ids,
                "notes": notes,
                "decomposed_raw": decomposed_notes,
            },
            f,
        )
