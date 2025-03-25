import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.model_selection import train_test_split

# 🚀 Disable Weights & Biases (W&B)
os.environ["WANDB_DISABLED"] = "true"

# 📌 Load Sample Dataset (Replace with your own CSV file)
data = pd.DataFrame({
    "text": [
        "I'm feeling really down today...", 
        "Life is beautiful!", 
        "I'm so anxious about my exams."
    ],
    "label": ["depressed", "neutral", "anxious"]  # Replace with actual labels
})

# ✅ Convert labels to numerical values
label_mapping = {"depressed": 0, "neutral": 1, "anxious": 2}
data["label"] = data["label"].map(label_mapping)

# 🔹 Split dataset into training & testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# 🔹 Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)

# 📌 Load tokenizer & model
model_name = "cardiffnlp/twitter-roberta-base-sentiment"  # Can be replaced with MentalRoBERTa if available
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)  # 3 sentiment classes

# ✅ Tokenization function
def tokenize_function(examples):
    tokenized_inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    tokenized_inputs["labels"] = examples["label"]  # Ensure labels are included
    return tokenized_inputs

# 🔹 Apply tokenization
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# 📌 Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",  # Disables W&B logging
)

# 🔹 Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# 🚀 Train the model
trainer.train()

# 🔥 Save the fine-tuned model
trainer.save_model("./mental_roberta_fine_tuned")

print("✅ MentalRoBERTa model fine-tuned and saved successfully!")
