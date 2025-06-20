from datasets import load_from_disk, concatenate_datasets, Dataset
import os
from collections import defaultdict
import math
import numpy as np


CHUNKS_DIR = "/Users/ritubansal/personal_projects/social_media_analysis/venv/dataset/processed_chunks"

# Get list of all chunk paths
chunk_dirs = sorted([
    os.path.join(CHUNKS_DIR, d)
    for d in os.listdir(CHUNKS_DIR)
    if d.startswith("chunk_") and os.path.isdir(os.path.join(CHUNKS_DIR, d))
])

# Load each chunk as a dataset
print(f"📂 Loading {len(chunk_dirs)} chunks...")
datasets = [load_from_disk(chunk_path) for chunk_path in chunk_dirs]

# Combine into one large dataset
print("Concatenating chunks...")
full_dataset = concatenate_datasets(datasets)

# Aggregation
# Step 1: Add day_block column
def map_day_block_batch(batch):
    hours = np.array(batch["hour"])
    day_blocks = np.full_like(hours, None, dtype=object)

    # Handle None values safely by masking
    mask_valid = hours != None
    valid_hours = hours[mask_valid].astype(int)

    blocks = np.full(valid_hours.shape, "Night", dtype=object)
    blocks[(valid_hours >= 23) | (valid_hours <= 4)] = "Late-Night"
    blocks[(valid_hours >= 5) & (valid_hours <= 8)] = "Early-Morning"
    blocks[(valid_hours >= 9) & (valid_hours <= 11)] = "Morning"
    blocks[(valid_hours >= 12) & (valid_hours <= 15)] = "Mid-day"
    blocks[(valid_hours >= 16) & (valid_hours <= 19)] = "Evening"

    day_blocks[mask_valid] = blocks
    return {"day_block": day_blocks.tolist()}

# Apply to full dataset
full_dataset = full_dataset.map(
    map_day_block_batch,
    batched=True,
    batch_size=5_000_000,
    num_proc=4
)

print("Added Day Block")

# Step 2: Aggregate using manual grouping

# STEP 1: Define batched aggregation map
def partial_aggregate(batch):
    partials = defaultdict(lambda: {"count": 0, "sentiment_sum": 0.0})

    for i in range(len(batch["sentiment"])):
        sentiment = batch["sentiment"][i]
        if sentiment is None or isinstance(sentiment, float) and math.isnan(sentiment):
            continue

        key = (
            batch["day_block"][i],
            batch["day"][i],
            batch["platform"][i],
            batch["main_emotion"][i],
            batch["primary_theme"][i],
            batch["secondary_theme_1"][i],
            batch["secondary_theme_2"][i],
            batch["secondary_theme_3"][i],
        )

        partials[key]["count"] += 1
        partials[key]["sentiment_sum"] += sentiment

    # Convert partials to lists
    keys, counts, sentiments = [], [], []
    for key, val in partials.items():
        keys.append(key)
        counts.append(val["count"])
        sentiments.append(val["sentiment_sum"])

    return {
        "key": keys,
        "count": counts,
        "sentiment_sum": sentiments
    }

# STEP 2: Run batched partial aggregation
partials_dataset = full_dataset.map(
    partial_aggregate,
    batched=True,
    batch_size=10_000,
    num_proc=4,
    remove_columns=full_dataset.column_names,
    desc="🔄 Partial aggregation"
)

# STEP 3: Reduce partial results to final grouped stats
final_agg = defaultdict(lambda: {"count": 0, "sentiment_sum": 0.0})

for row in partials_dataset:
    key = tuple(row["key"])
    final_agg[key]["count"] += row["count"]
    final_agg[key]["sentiment_sum"] += row["sentiment_sum"]

# STEP 4: Format output
grouped_data = []

for key, val in final_agg.items():
    grouped_data.append({
        "day_block": key[0],
        "day": key[1],
        "platform": key[2],
        "main_emotion": key[3],
        "primary_theme": key[4],
        "secondary_theme_1": key[5],
        "secondary_theme_2": key[6],
        "secondary_theme_3": key[7],
        "count": val["count"],
        "avg_sentiment": val["sentiment_sum"] / val["count"] if val["count"] > 0 else None
    })

grouped_dataset = Dataset.from_list(grouped_data)
print("Grouped aggregation complete.")

# Step 5: Save to disk
OUTPUT_PATH = "/Users/ritubansal/personal_projects/social_media_analysis/venv/dataset/grouped_data"

grouped_dataset.save_to_disk(OUTPUT_PATH)

print(f"Saved grouped dataset to {OUTPUT_PATH}")