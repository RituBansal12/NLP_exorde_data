import os
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from datasets import load_from_disk
from scipy.stats import entropy

# === CONFIGURATION ===
INPUT_DIR = "/Users/ritubansal/personal_projects/social_media_analysis/venv/dataset/filtered_data"
OUTPUT_PATH = "/Users/ritubansal/personal_projects/social_media_analysis/venv/dataset/final_aggregated_mentions.json"

# === PERSON MATCH MAP ===
PEOPLE_MATCH_MAP = {
    "donald trump": ["trump", "donald", "donald trump", "donald j trump", "djt", "president trump"],
    "elon musk": ["elon", "musk", "elonmusk", "elon musk"],
    "kamala harris": ["kamala", "harris", "vp harris", "kamala harris"],
    "joe biden": ["biden", "joe", "joe biden", "president biden", "joebiden"],
    "jd vance": ["jd", "vance", "jd vance"]
}
alias_groups = {person: set(aliases) for person, aliases in PEOPLE_MATCH_MAP.items()}

# === ENTROPY CALCULATION FUNCTION ===
def calculate_sentiment_entropy(sentiments, bins=10):
    """
    Calculate normalized entropy of sentiment distribution.
    
    Args:
        sentiments: array of sentiment values
        bins: number of bins to discretize continuous sentiment values
    
    Returns:
        normalized entropy value between 0-1 (higher = more uncertain/random distribution)
    """
    if len(sentiments) == 0:
        return 0.0
    
    # Create histogram to discretize continuous sentiment values
    hist, _ = np.histogram(sentiments, bins=bins, density=True)
    
    # Remove zero bins to avoid log(0)
    hist = hist[hist > 0]
    
    # Normalize to get probabilities
    probs = hist / np.sum(hist)
    
    # Calculate entropy using scipy
    ent = entropy(probs, base=2)
    
    # Normalize by maximum possible entropy for uniform distribution
    max_entropy = np.log2(bins)
    
    return ent / max_entropy if max_entropy > 0 else 0.0

# === MENTION DETECTION FUNCTION ===
def detect_mentions(example):
    keyword_str = example.get("english_keywords", "")
    # Convert to list of cleaned lowercase keywords
    keywords = set(kw.strip().lower() for kw in keyword_str.split(",") if kw.strip())

    for person, aliases in alias_groups.items():
        example[f"mentions_{person.replace(' ', '_')}"] = bool(aliases & keywords)
    return example

# === SENTIMENT AGGREGATION ===
sentiment_data = defaultdict(list)  # key: person, value: list of sentiments

# === MAIN SCRIPT ===
def main():
    print("Processing chunks...")
    for fname in tqdm(sorted(os.listdir(INPUT_DIR))):
        path = os.path.join(INPUT_DIR, fname)
        if os.path.isdir(path):
            try:
                print(f"\nLoading chunk: {fname}")
                chunk = load_from_disk(path)
                chunk = chunk.remove_columns(
                    [col for col in chunk.column_names if col not in ["english_keywords", "sentiment"]]
                )

                print(f"Detecting mentions...")
                chunk = chunk.map(detect_mentions, desc=f"Detecting mentions in {fname}")

                # Show one sample row with mention flags
                print("First row after mention detection:")
                print(chunk[0])

                # Aggregate per person
                for person in alias_groups:
                    col = f"mentions_{person.replace(' ', '_')}"
                    print(f"Filtering for mentions of: {person}")
                    filtered = chunk.filter(lambda x: x[col], desc=f"Filtering {person}")

                    print(f"Found {len(filtered)} rows for {person}")
                    if len(filtered) > 0:
                        print("First filtered row:", filtered[0])
                        sentiment_data[person].extend(filtered["sentiment"])

                print(f"Finished {fname}")

            except Exception as e:
                print(f"Skipped {fname} due to error: {e}")

    print("\nFinal aggregation...")

    # Compute final stats including entropy
    output = []
    for person, sentiments in sentiment_data.items():
        if not sentiments:
            continue
        sentiments = np.array(sentiments)
        
        # Calculate normalized entropy
        normalized_entropy = calculate_sentiment_entropy(sentiments, bins=10)
        
        output.append({
            "person": person.title(),
            "count": len(sentiments),
            "avg_sentiment": float(np.mean(sentiments)),
            "min_sentiment": float(np.min(sentiments)),
            "max_sentiment": float(np.max(sentiments)),
            "median_sentiment": float(np.median(sentiments)),
            "std_sentiment": float(np.std(sentiments)),
            "percentile_10": float(np.percentile(sentiments, 10)),
            "percentile_25": float(np.percentile(sentiments, 25)),
            "percentile_75": float(np.percentile(sentiments, 75)),
            "percentile_90": float(np.percentile(sentiments, 90)),
            "sentiment_entropy": float(normalized_entropy)
        })

    # Sort by count
    output.sort(key=lambda x: x["count"], reverse=True)

    # Save
    print(f"Saving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print("Done!")

    # Print first line of final output
    if output:
        print("\nFirst line of final aggregated mentions:")
        print(json.dumps(output[0], indent=2))
        
        # Print entropy summary
        print("\nEntropy Summary:")
        for item in output:
            print(f"{item['person']}: Entropy = {item['sentiment_entropy']:.3f}")
    else:
        print("No mentions found in any chunks.")

if __name__ == "__main__":
    main()






