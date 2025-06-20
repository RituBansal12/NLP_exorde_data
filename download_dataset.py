from datasets import load_dataset

# Replace 'imdb' with your dataset ID
dataset = load_dataset("Exorde/exorde-social-media-one-month-2024")

# Save to local folder
dataset.save_to_disk("/Users/ritubansal/personal_projects/social_media_analysis/venv/dataset/social_media_data")

print("Download complete!")