from datasets import load_dataset, Dataset
from itertools import islice
from tqdm import tqdm
import os

# === CONFIGURATION ===
DATASET_ID = "Exorde/exorde-social-media-one-month-2024"
OUTPUT_DIR = "/Users/ritubansal/personal_projects/social_media_analysis/venv/dataset/filtered_data"
CHUNK_SIZE = 1_000_000  # Save in manageable chunks
MAX_ROWS = 20_000_000  # Stop matching after 20M results found

os.makedirs(OUTPUT_DIR, exist_ok=True)

def stream_and_filter():
    dataset = load_dataset(DATASET_ID, split="train", streaming=True)
    chunk = []
    total = 0
    chunk_id = 0

    print("🌍 Streaming and filtering dataset (language='en', primary_theme='Politics')...")

    for post in tqdm(dataset, desc="🌀 Streaming"):
        if post.get("language") != "en":
            continue
        if post.get("primary_theme") != "Politics":
            continue

        chunk.append(post)
        total += 1

        # Save every CHUNK_SIZE rows
        if len(chunk) >= CHUNK_SIZE:
            out_path = os.path.join(OUTPUT_DIR, f"chunk_{chunk_id}")
            Dataset.from_list(chunk).save_to_disk(out_path)
            print(f"✅ Saved chunk {chunk_id} with {len(chunk):,} rows.")
            chunk = []
            chunk_id += 1

        if total >= MAX_ROWS:
            break

    # Save any remaining data
    if chunk:
        out_path = os.path.join(OUTPUT_DIR, f"chunk_{chunk_id}")
        Dataset.from_list(chunk).save_to_disk(out_path)
        print(f"✅ Saved final chunk {chunk_id} with {len(chunk):,} rows.")

    print(f"\n🎉 Done! Total filtered posts saved: {total:,}")

# === RUN ===
if __name__ == "__main__":
    stream_and_filter()
