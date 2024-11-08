import json

# Load your dataset
input_file = "TCM_Dataset_Sample.json"
output_file = "ChatGLM4_formatted_dataset.jsonl"

# transform the dataset from JSONL format
def transform_data(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            # Parse each line as a JSON object
            entry = json.loads(line.strip())
            # Reformat to the ChatGLM-4 format
            formatted_entry = {
                "messages": [
                    {"role": "user", "content": entry["query"]},
                    {"role": "assistant", "content": entry["response"]}
                ]
            }
            # Write each reformatted entry as a line in JSONL format
            json.dump(formatted_entry, f_out, ensure_ascii=False)
            f_out.write("\n")

# Run the transformation function
transform_data(input_file, output_file)