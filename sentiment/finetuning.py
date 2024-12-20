# 가장 먼저, finetunig을 통해 영화 리뷰도 긍정, 부정을 분류할 수 있는 모델을 만들도록 하겠습니다.
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from datasets import load_metric
from transformers import TrainingArguments, Trainer
from evaluate import load as load_evaluate_metric 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imdb = load_dataset("imdb")

# 데이터셋 불러오기 (작은 사이즈로 훈련시킬게요)
small_train_dataset = imdb["train"].shuffle(seed=42).select([i for i in list(range(10000))])
small_test_dataset = imdb["test"].shuffle(seed=42).select([i for i in list(range(1000))])

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
   return tokenizer(examples["text"], truncation=True)
 
tokenized_train = small_train_dataset.map(preprocess_function, batched=True)
tokenized_test = small_test_dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model.to(device)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy_metric = load_evaluate_metric("accuracy")
    f1_metric = load_evaluate_metric("f1")
    
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=predictions, references=labels)["f1"]
    
    return {"accuracy": accuracy, "f1": f1}


repo_name = "finetuning-sentiment-model-10000-samples"

training_args = TrainingArguments(
   output_dir=repo_name,
   learning_rate=2e-5,
   per_device_train_batch_size=16,
   per_device_eval_batch_size=16,
   num_train_epochs=3,
   weight_decay=0.01,
   save_strategy="epoch",
   push_to_hub=True,
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

trainer.train()

trainer.evaluate()

trainer.push_to_hub()


