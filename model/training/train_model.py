#!/usr/bin/env python3
"""
Train script for fine-tuning CodeBERT on compilation error fixing task.
"""

import json
import torch
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, 
                         DataCollatorForSeq2Seq, Seq2SeqTrainingArguments,
                         Seq2SeqTrainer)
from datasets import Dataset, load_from_disk
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    """Load and preprocess the training data"""
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Format input: error message + context code
    formatted_data = []
    for item in data:
        input_text = f"Fix this Java error: {item['error_message']} \n Code: {item['context_code']}"
        target_text = item['diff_patch']
        formatted_data.append({"input": input_text, "target": target_text})
    
    return formatted_data

def main():
    # Load dataset
    print("Loading and preprocessing data...")
    train_data = load_and_preprocess_data('data/processed/train_dataset.jsonl')
    val_data = load_and_preprocess_data('data/processed/val_dataset.jsonl')
    
    # Convert to Hugging Face dataset
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    # Load tokenizer and model
    print("Loading CodeBERT model and tokenizer...")
    model_name = "microsoft/codebert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Tokenization function
    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["input"],
            max_length=512,
            truncation=True,
            padding="max_length"
        )
        
        labels = tokenizer(
            examples["target"], 
            max_length=256,
            truncation=True,
            padding="max_length"
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # Tokenize datasets
    print("Tokenizing data...")
    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./codebert-fix-model",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=5,
        predict_with_generate=True,
        fp16=True,
        logging_dir="./logs",
        logging_steps=10,
        report_to="tensorboard"
    )
    
    # Create trainer
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save final model
    trainer.save_model("./codebert-fix-model-final")
    tokenizer.save_pretrained("./codebert-fix-model-final")
    
    print("Training completed and model saved!")

if __name__ == "__main__":
    main()