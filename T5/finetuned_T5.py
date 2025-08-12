import pandas as pd
import numpy as np
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, TrainerCallback
from datasets import Dataset, DatasetDict
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm

instruction = '''Your task is to extract the following categories from the input text:
SuperGroup, Group, module

The input text will already be provided in the form: SuperGroup,Group,module.
Return the output exactly in this format:

SuperGroup: <supergroup> Group: <group> module: <module>
'''

train_data = pd.read_csv('data/train_data.csv')
test_data = pd.read_csv('data/test_data.csv')
val_data = pd.read_csv('data/val_data.csv')

def preprocess_data(data):
    inputs = []
    targets = []
    for _, row in data.iterrows():
        input_parts = [instruction]
        if 'retailer' in row and pd.notna(row['retailer']):
            input_parts.append(f"retailer: {row['retailer']}")
        if 'price' in row and pd.notna(row['price']):
            input_parts.append(f"price: {row['price']}")
        if 'Description' in row and pd.notna(row['Description']):
            input_parts.append(f"Description: {row['Description']}")            
        input_text = "\n".join(input_parts)

        supergroup = row.get('supergroup') 
        group = row.get('group') 
        module = row.get('module') 

        target_text = f"SuperGroup: {supergroup} Group: {group} module: {module}"
        inputs.append(input_text)
        targets.append(target_text)

    return pd.DataFrame({'input_text': inputs, 'target_text': targets})

train_df = preprocess_data(train_data)
val_df = preprocess_data(val_data)
test_df = preprocess_data(test_data)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

dataset_dict = DatasetDict({'train': train_dataset, 'validation': val_dataset, 'test': test_dataset})

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

def preprocess_function(examples):
    inputs = examples['input_text']
    targets = examples['target_text']
    model_inputs = tokenizer(inputs, max_length=512, padding='max_length', truncation=True)
    labels = tokenizer(targets, max_length=64, padding='max_length', truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized = dataset_dict.map(preprocess_function, batched=True, remove_columns=['input_text','target_text'])

training_args = TrainingArguments(
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['validation'],
)

trainer.train()

trainer.save_model('./fine_tuned_t5')
tokenizer.save_pretrained('./fine_tuned_t5')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = T5ForConditionalGeneration.from_pretrained('./fine_tuned_t5').to(device)
tokenizer = T5Tokenizer.from_pretrained('./fine_tuned_t5')
model.eval()

def generate_text(input_text, max_length=64):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def extract_categories(text):
    """
    Expect: SuperGroup: <...> Group: <...> module: <...>
    Returns (supergroup, group, module) or ('na','na','na')
    """
    pattern = r'SuperGroup:\s*(.*?)\s+Group:\s*(.*?)\s+module:\s*(.*)$'
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        return tuple(item.strip() if item and item.strip() != '' else 'na' for item in match.groups())
    # fallback: split by comma/pipe/semicolon into 3 parts
    parts = [p.strip() for p in re.split(r'[,\|;]', text) if p.strip()]
    if len(parts) >= 3:
        return (parts[0], parts[1], parts[2])
    return ('na', 'na', 'na')

test_inputs = test_df['input_text'].tolist()
test_targets = test_df['target_text'].tolist()

total = 0
correct_all_three = 0
results = []  

for inp, tgt in tqdm(zip(test_inputs, test_targets), total=len(test_inputs), desc="Evaluating"):
    pred = generate_text(inp)
    pred_cats = extract_categories(pred)
    tgt_cats = extract_categories(tgt)
    is_match = (pred_cats == tgt_cats)
    total += 1
    if is_match:
        correct_all_three += 1

exact_match_accuracy = correct_all_three / total if total > 0 else 0.0
print(f"Accuracy : {exact_match_accuracy:.4f} ({correct_all_three}/{total})")

