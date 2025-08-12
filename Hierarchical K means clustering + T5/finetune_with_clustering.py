import pandas as pd
import numpy as np
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, TrainerCallback
from datasets import Dataset, DatasetDict
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm

train_data = pd.read_csv('data/train_data.csv')
test_data = pd.read_csv('data/test_data.csv')
val_data = pd.read_csv('data/val_data.csv')

instruction = '''Your task: from the input context, predict the CODE string corresponding to the module.
Input will contain retailer/price/Description and the module is implied in that context.
Return the output as the code string exactly, e.g.:
1, 1, 0, 5, 0
(only the code string; no extra labels)
'''
mod_code_df = pd.read_csv('data/mod_code.csv')
mod_code_df['module_name'] = mod_code_df['module_name'].astype(str).str.strip()
mod_code_df['code'] = mod_code_df['code'].astype(str).str.replace(r'\s*,\s*', ', ', regex=True).str.strip()
# module_name -> code
module_to_code = dict(zip(mod_code_df['module_name'], mod_code_df['code']))
# code -> module_name
code_to_module=dict(zip(mod_code_df['code'], mod_code_df['module_name']))
# module -> [group,supergroup]
module_to_group_super = {}
all_dfs = pd.concat([train_data, val_data, test_data], ignore_index=True)
for _, row in all_dfs.iterrows():
    module = row.get('module')
    group = row.get('group') 
    supergroup = row.get('supergroup') 
    module = str(module).strip()
    group = str(group).strip() if pd.notna(group) else 'na'
    supergroup = str(supergroup).strip() if pd.notna(supergroup) else 'na'
    module_to_group_super[module] = [group, supergroup]
        
def preprocess_data(data):
    inputs = []
    targets = []
    for _, row in data.iterrows():
        input_parts = []
        if 'retailer' in row and pd.notna(row['retailer']):
            input_parts.append(f"retailer: {row['retailer']}")
        if 'price' in row and pd.notna(row['price']):
            input_parts.append(f"price: {row['price']}")
        if 'Description' in row and pd.notna(row['Description']):
            input_parts.append(f"Description: {row['Description']}")            
        input_text = "\n".join(input_parts)
        target_text = module_to_code.get(row['module'])
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

def normalize_code(code_str):
    if code_str is None:
        return None
    s = str(code_str).strip()
    # unify commas and spacing: '1,1,0' -> '1, 1, 0'
    s = re.sub(r'\s*,\s*', ', ', s)
    return s

test_inputs = test_df['input_text'].tolist()
test_targets = test_df['target_text'].tolist()

total = 0
correct = 0
results = []

for inp, tgt_code in tqdm(zip(test_inputs, test_targets), total=len(test_inputs), desc="Evaluating"):
    pred_code_raw = generate_text(inp)
    pred_code = normalize_code(pred_code_raw)
    tgt_code_norm = normalize_code(tgt_code)

    total += 1
    tgt_module = code_to_module.get(tgt_code_norm)
    tgt_group, tgt_super = module_to_group_super.get(tgt_module, (None, None))    
    pred_module = code_to_module.get(pred_code)  # None if predicted code not in mapping
    pred_group = pred_super = None
    if pred_module:
        pred_group, pred_super = module_to_group_super.get(pred_module, (None, None))
    is_correct = False
    if pred_module is not None:
        if (pred_module == tgt_module):
            is_correct = True
    if is_correct:
        correct += 1

accuracy = correct / total if total > 0 else 0.0
print(f"Accuracy : {accuracy:.4f} ({correct}/{total})")

