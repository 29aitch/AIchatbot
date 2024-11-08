import json
import random

# Load the dataset
file_path = '../ChatGLM4_formatted_dataset.jsonl'
dataset = []

with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        dataset.append(json.loads(line))

# Shuffle dataset for random splitting
random.shuffle(dataset)

# Define split ratios
train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
total_size = len(dataset)

# Calculate split sizes
train_size = int(total_size * train_ratio)
val_size = int(total_size * val_ratio)

# Split dataset
train_data = dataset[:train_size]
val_data = dataset[train_size:train_size + val_size]
test_data = dataset[train_size + val_size:]

# Save the splits to separate JSONL files
split_paths = {
    "train": "Dataset/Splitted_Datasets/ChatGLM4_train.jsonl",
    "validation": "Dataset/Splitted_Datasets/ChatGLM4_validation.jsonl",
    "test": "Dataset/Splitted_Datasets/ChatGLM4_test.jsonl"
}

for split_name, data in zip(split_paths.keys(), [train_data, val_data, test_data]):
    with open(split_paths[split_name], 'w', encoding='utf-8') as split_file:
        for entry in data:
            split_file.write(json.dumps(entry, ensure_ascii=False) + '\n')

print("Dataset split completed.")
