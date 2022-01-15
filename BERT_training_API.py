import sys
import time

import torch as torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

from BERT_training import tokenizer, out_label_ids

print('Python version     : ' + str(sys.version))
print('Torch version   : ' + str(torch.__version__))

# instantiating from the pre-trained model gbert for German language
#config = BertConfig.from_pretrained("deepset/gbert-base", finetuning_task='binary', num_labels=2)
#tokenizer = BertTokenizer.from_pretrained("deepset/gbert-base")
model = AutoModelForSequenceClassification.from_pretrained("deepset/gbert-base", num_labels=2)

# loading prepared data
df = pd.read_csv(r'Prepared_data.csv', engine='python', sep='delimiter', encoding='ISO-8859-1',
                 delimiter=';', header=0)

# lowercase tokens (do i need it?)
df['text'] = df['text'].str.lower()

# renaming the column hof_OR_none to labels, inplace updates original object
df.rename(columns={'hof_OR_none': 'labels'}, inplace=True)

# print(df.head(5))

# converting labels into numerical values, in order to save it into a tensor
df['labels'] = df['labels'].replace(['HOF', 'NOT'], [1, 0])

# check new data type of column labels: should be int now
# print(df.labels.dtypes)

# check data: print(df['labels'].sample(15))

# DATA SPLIT TO TRAINING AND TEST
# size 20 to 80
X_train, X_test, Y_train, Y_test = train_test_split(df['text'],
                                                    df['labels'],
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=df['labels']
                                                    )

# TOKENIZATION

# tokenize training set
X_train_tokens = []
for row in X_train:
    X_train_tokens.append(tokenizer.encode(row,
                                           add_special_tokens=True,
                                           max_length=50,
                                           truncation=True,
                                           padding='max_length'
                                           ))

# creating the attention mask for training data
att_mask_train = [[float(id > 0) for id in seq] for seq in X_train_tokens]

# tokenize test set
X_test_tokens = []
for row in X_test:
    X_test_tokens.append(tokenizer.encode(row,
                                          add_special_tokens=True,
                                          max_length=50,
                                          truncation=True,
                                          padding='max_length',
                                          # return_tensors='pt'  # returns tensors
                                          ))

# creating the attention mask for test data
att_mask_test = [[float(id > 0) for id in seq] for seq in X_test_tokens]

# CURRENT DATA TYPES
print(att_mask_train[0])
print(type(X_train_tokens))
print(type(att_mask_train))
print(type(Y_train))
# print(len(X_train_tokens[2]))

# Create Tensors for training set, attention mask, and training labels
input_ids_train = torch.LongTensor(X_train_tokens)
input_mask_train = torch.FloatTensor(att_mask_train)
label_ids_train = torch.tensor(Y_train.values)

# Test Data Tensors
ids_test_data = torch.LongTensor(X_test_tokens)
input_mask_test = torch.FloatTensor(att_mask_test)
label_ids_test = torch.tensor(Y_test.values)


# shapes of this tensors
print('\n------------------------')
print(input_ids_train.shape)
print(input_mask_train.shape)
print(label_ids_train.shape)
print('------------------------')

# input example
print('sample input tokens: ' + str(input_ids_train[1]))
print('attention mask: ' + str(input_mask_train[1]))
print('label: ' + str(label_ids_train[1]))

# Concatenate tensors into one tensor
training_data = TensorDataset(input_ids_train, input_mask_train, label_ids_train)
test_data = TensorDataset(ids_test_data, input_mask_test, label_ids_test)

# MODEL TRAINING
print('-----------------------')

training_args = TrainingArguments("test_trainer",
                                  evaluation_strategy="epoch",
                                  learning_reate=2e-5,
                                  per_device_train_batch_size=16,
                                  per_device_eval_batch_size=16,
                                  num_train_epochs=4,
                                  weight_decay=0.01,
                                  )

#trainer for training
trainer = Trainer(model=model,
                  args=training_data,
                  train_dataset=training_data,
                  eval_dataset=test_data
                  )

trainer.train()

# metric = accuracy_score(preds, out_label_ids)

#def compute_metrics(eval_pred):
    #logits, labels = eval_pred
    #predictions = np.argmax(logits, axis=-1)
    #return metric.compute(predictions=predictions, references=labels)

#trainer for evaluation
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_data,
    eval_dataset=test_data,
    compute_metrics=accuracy_score
)

trainer.evaluate()
