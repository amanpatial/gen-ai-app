import json
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("tomshe/Internal_Medicine_questions_binary")

print("Original sample:")
print(ds['train'][0])
print()

# Create a function to transform sample to keep only message and answer
def transform_sample(sample):
    message = f"Question: {sample['question']}\nOption A: {sample['optionA']}\nOption B: {sample['optionB']}"
    return {
        'message': message,
        'answer_idx': sample['answer idx']
    }

# Apply the transformation to the dataset
ds = ds.map(transform_sample, remove_columns=['question', 'optionA', 'optionB', 'answer idx', 'answer'])

# Shuffle the dataset
ds = ds.shuffle(seed=42)

# Select 50 samples
ds_subset = ds['train'].select(range(min(50, len(ds['train']))))

print(f"Selected {len(ds_subset)} samples")
print("Sample transformed entry:")
print(ds_subset[0])
print()

# Write to JSONL file in OpenAI format
with open('medical_questions.jsonl', 'w', encoding='utf-8') as f:
    for sample in ds_subset:
        json.dump({"item": sample}, f, ensure_ascii=False)
        f.write('\n')

print("Dataset saved to medical_questions.jsonl")