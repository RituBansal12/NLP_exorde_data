from datasets import load_from_disk, concatenate_datasets
from urllib.parse import urlparse
from datetime import datetime
import os

# === CONFIGURATION ===
DATASET_PATH = "/Users/ritubansal/personal_projects/social_media_analysis/venv/dataset/social_media_data"
OUTPUT_DIR = "/Users/ritubansal/personal_projects/social_media_analysis/venv/dataset/processed_chunks"
CHUNK_SIZE = 5_000_000 
NUM_PROC = 4  # Adjust based on your Mac’s cores

# === THEME MAPPING ===
theme_mapping = {
    1: "Economy", 2: "Technology", 3: "Investing", 4: "Business", 5: "Cryptocurrency",
    6: "Social", 7: "Politics", 8: "Finance", 9: "Entertainment", 10: "Health",
    11: "Law", 12: "Sports", 13: "Science", 14: "Environment", 15: "People"
}

def preprocess_batch(batch):
    platforms, hours, days = [], [], []
    st1, st2, st3 = [], [], []

    for i in range(len(batch["url"])):
        domain = urlparse(batch["url"][i]).netloc or ""
        platforms.append(domain)

        try:
            dt = datetime.strptime(batch["date"][i], "%Y-%m-%dT%H:%M:%S.%fZ")
            hours.append(dt.hour)
            days.append(dt.strftime("%a"))
        except:
            hours.append(None)
            days.append(None)

        sec_themes = batch["secondary_themes"][i] or []
        mapped = [theme_mapping.get(x, "Unknown") for x in sec_themes]
        mapped += [None] * (3 - len(mapped))
        st1.append(mapped[0])
        st2.append(mapped[1])
        st3.append(mapped[2])

    return {
        "platform": platforms,
        "hour": hours,
        "day": days,
        "secondary_theme_1": st1,
        "secondary_theme_2": st2,
        "secondary_theme_3": st3,
    }

# === STEP 1: LOAD + FILTER ENGLISH ===
print("Loading dataset and filtering English rows...")
dataset = load_from_disk(DATASET_PATH)
english_dataset = dataset["train"].filter(lambda x: x["language"] == "en", num_proc=NUM_PROC)

# === STEP 2: CHUNK AND PROCESS ===
total_rows = len(english_dataset)
num_chunks = (total_rows // CHUNK_SIZE) + 1

print(f"Processing {total_rows:,} rows in {num_chunks} chunks...")

os.makedirs(OUTPUT_DIR, exist_ok=True)

for i in range(num_chunks):
    start = i * CHUNK_SIZE
    end = min((i + 1) * CHUNK_SIZE, total_rows)

    print(f"Chunk {i + 1}/{num_chunks} → rows {start:,} to {end:,}")

    chunk = english_dataset.select(range(start, end))

    processed_chunk = chunk.map(
        preprocess_batch,
        batched=True,
        batch_size=1000,
        num_proc=NUM_PROC,
        desc=f"Preprocessing chunk {i + 1}"
    )

    final_chunk = processed_chunk.remove_columns([
        col for col in processed_chunk.column_names if col not in [
            "hour", "day", "platform", "primary_theme", "sentiment", "main_emotion",
            "secondary_theme_1", "secondary_theme_2", "secondary_theme_3"
        ]
    ])

    output_path = os.path.join(OUTPUT_DIR, f"chunk_{i}")
    final_chunk.save_to_disk(output_path)

print("All chunks processed and saved successfully.")
