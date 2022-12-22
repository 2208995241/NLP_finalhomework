from transformers import AutoTokenizer,AutoModel,DataCollatorWithPadding,TrainingArguments,AutoModelForSequenceClassification,Trainer
from datasets import load_dataset,load_metric
import numpy as np
import torch
#checkpoint = "bert-base-uncased"
checkpoint = "fnlp/bart-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
raw_puts = ["hide new secretions from the parental units",
            "contains no wit , only labored gags "]
inputs = tokenizer(raw_puts,padding=True,truncation=True,return_tensors="pt")
print(inputs)
#model = AutoModel.from_pretrained(checkpoint)
raw_datasets = load_dataset("glue","sst2")
print(raw_datasets)
raw_train_dataset = raw_datasets["train"]
print(raw_train_dataset.features)
def tokenize_function(example):
    return tokenizer(example["sentence"],truncation=True)
tokenized_datasets = raw_datasets.map(tokenize_function,batched=True)
print(tokenized_datasets)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
print(tokenized_datasets["train"])
samples = tokenized_datasets["train"][:8]
samples = {k:v for k,v in samples.items() if k not in["idx","sentence"]}
print(len(x) for x in samples["input_ids"])
batch = data_collator(samples)
print({k:v.shape for k,v in batch.items()})
training_args = TrainingArguments(output_dir="test-trainer-bert-base",num_train_epochs=30,adam_epsilon=1e-1)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint,num_labels=2)
optimizer = torch.optim.Adam(model.parameters(),lr=5e-5)
small_train_dataset = tokenized_datasets["train"].select(range(50))
small_eval_dataset = tokenized_datasets["validation"].select(range(50))

def compute_metrics(eval_preds):
    metric = load_metric("glue","sst2")
    logits,labels = eval_preds
    predictions = np.argmax(logits,axis=-1)
    return metric.compute(prediction =predictions,references=labels)
trainer = Trainer(model,training_args,train_dataset=small_train_dataset
                  ,eval_dataset=small_eval_dataset,
                  data_collator=data_collator,
                  tokenizer=tokenizer,
                  compute_metrics=compute_metrics,
                  #optimizers=optimizer,
                  )
print(training_args)
trainer.train()