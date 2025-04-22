import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
import pandas as pd
import numpy as np
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class BertPredictor:

    def __init__(self, random_state=42, device="cuda:0", model_ckp=None, no_refit=False):
        # Random state is not used

        self.device = device
        self.no_refit = no_refit

        self.tokenizer = AutoTokenizer.from_pretrained(model_ckp)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_ckp,
            num_labels=1,
            problem_type="regression"
        ).to('cpu') # Load model to CPU to avoid CUDA memory usage
        self.index = 0

    def fit(self, X, y):
        if self.no_refit:
            return
        archs = X.loc[:,0].tolist()
        df = pd.DataFrame({"arch": archs, "labels": torch.tensor(y)})
        train_ds = Dataset.from_pandas(df)

        def _tokenize_function(examples):
            return self.tokenizer(examples["arch"], padding='longest', truncation=True, return_tensors="pt")
        train_ds = train_ds.map(_tokenize_function, batched=True, drop_last_batch=True, batch_size=2)

        self.model.to(self.device) # Load model to CUDA
        self.model.train()

        train_args = TrainingArguments(
            output_dir="/tmp/trainer_output",
            do_eval=False,
            learning_rate=1e-5,
            per_device_train_batch_size=1,
            num_train_epochs=2,
            weight_decay=0.01,
            warmup_ratio=0.06,
            logging_steps=100,
            lr_scheduler_type="cosine",
            push_to_hub=False,
            report_to="none",
            save_strategy="no",
        )
        trainer = Trainer(
            model=self.model,
            args=train_args,
            train_dataset=train_ds,
            processing_class=self.tokenizer
        )
        trainer.train()

        self.model.eval()
        self.model.to('cpu') # Load model to CPU to avoid CUDA memory usage
        torch.cuda.empty_cache()

    def _preprocess_func(self, arch_str):
        return self.tokenizer(arch_str, 
                              padding='longest', 
                              return_tensors='pt', 
                              truncation=True).to(self.device)
    
    def predict(self, X):
        self.model.to(self.device) # Load model to CUDA
        self.model.eval()

        inputs = self._preprocess_func(X.loc[:,0].tolist())
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.detach().cpu().numpy().tolist()
            self.index += 1

        self.model.to('cpu') # Load model to CPU to avoid CUDA memory usage
        torch.cuda.empty_cache()
        return logits