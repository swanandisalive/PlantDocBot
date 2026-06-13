import pandas as pd
import numpy as np
import torch
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
import evaluate
import joblib
import shutil
import os

# ==========================================
# 1. LOAD AND PREPARE DATA
# ==========================================
print("⬇️  Loading Dataset...")
ds = load_dataset("ButterChicken98/plantvillage-image-text-pairs")
df = pd.DataFrame(ds['train'])

# Drop image column to save memory
if 'image' in df.columns:
    df = df.drop(columns=['image'])

# Rename caption to text
if 'caption' in df.columns:
    df = df.rename(columns={'caption': 'text'})

print(f"✅ Data Loaded: {len(df)} rows")

# ==========================================
# 2. LABEL ENCODING
# ==========================================
print("🏷️  Encoding Labels...")
le = LabelEncoder()

# Create labels based on keywords if 'label' column is missing
if 'label' not in df.columns:
    df['label'] = df['text'].apply(lambda x: 'Healthy' if 'healthy' in str(x).lower() else 'Disease')

# Create the numeric label column
df['label_id'] = le.fit_transform(df['label'])

# Save label encoder
joblib.dump(le, "label_encoder.pkl")
print(f"   Classes found: {le.classes_}")

# ==========================================
# 3. SPLIT DATA
# ==========================================
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# ==========================================
# 4. TOKENIZATION
# ==========================================
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

print("🔠 Tokenizing data...")
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# --- CRITICAL FIX: CLEAN COLUMNS FOR TRAINER ---
print("🧹 Cleaning columns for Trainer...")

# 1. Identify columns to remove (anything that isn't input_ids, attention_mask, or labels)
# We keep 'label_id' for a moment to rename it.
cols_to_remove = [col for col in tokenized_train.column_names if col not in ["input_ids", "attention_mask", "label_id"]]
tokenized_train = tokenized_train.remove_columns(cols_to_remove)
tokenized_test = tokenized_test.remove_columns(cols_to_remove)

# 2. Rename 'label_id' to 'labels' (Hugging Face expects exactly the name 'labels')
tokenized_train = tokenized_train.rename_column("label_id", "labels")
tokenized_test = tokenized_test.rename_column("label_id", "labels")

# 3. Set format to PyTorch tensors explicitly
tokenized_train.set_format("torch")
tokenized_test.set_format("torch")
# -----------------------------------------------

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ==========================================
# 5. LOAD MODEL
# ==========================================
num_labels = len(le.classes_)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# ==========================================
# 6. METRICS
# ==========================================
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# ==========================================
# 7. TRAINING
# ==========================================
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch", 
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("🚀 Starting Training (This may take 5-10 minutes)...")
trainer.train()

# ==========================================
# 8. SAVING
# ==========================================
save_path = "./plant_bert_model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"✅ Model saved to {save_path}")

try:
    shutil.make_archive("plant_bert_model", 'zip', save_path)
    print("📦 Model zipped as plant_bert_model.zip")
except Exception as e:
    print(f"Could not zip file: {e}")