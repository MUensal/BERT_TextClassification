import sys
import time

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


print('Python version     : ' + str(sys.version))
print('Torch version   : ' + str(torch.__version__))

# instantiating from the pre-trained model gbert for German language
config = BertConfig.from_pretrained("deepset/gbert-base", finetuning_task='binary', num_labels=2)
tokenizer = BertTokenizer.from_pretrained("deepset/gbert-base")
model = BertForSequenceClassification.from_pretrained("deepset/gbert-base")

# loading prepared data
df = pd.read_csv(r'Prepared_data_shorter.csv', engine='python', sep='delimiter', encoding='ISO-8859-1',
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

# print(X_train[:10])
# print(X_train.index)


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
#print(att_mask_train[0])
print(type(X_train_tokens))
print(type(att_mask_train))
print(type(Y_train))
# print(len(X_train_tokens[2]))

# Create Tensors for training set, attention mask, and training labels
training_data_x = torch.LongTensor(X_train_tokens)
input_mask_train = torch.FloatTensor(att_mask_train)
training_labels_y = torch.tensor(Y_train.values)

# Test Data Tensors
test_data_x = torch.LongTensor(X_test_tokens)
input_mask_test = torch.FloatTensor(att_mask_test)
test_labels_y = torch.tensor(Y_test.values)

print("\n-------------INFORMATION--------------")
print(np.shape(training_data_x))
print(np.shape(training_labels_y))
print(np.shape(test_data_x))
print(np.shape(test_labels_y))
print(type(training_data_x))
print(test_labels_y[:3])
print(training_labels_y[:3])
print(len(test_labels_y))
print(len(training_labels_y))
print(type(X_train))


# shapes of this tensors
print('\n------------------------')
print(training_data_x.shape)
print(input_mask_train.shape)
print(training_labels_y.shape)
print('------------------------')

# input example
print('sample input tokens: ' + str(training_data_x[2]))
print('attention mask: ' + str(input_mask_train[2]))
print('label: ' + str(training_labels_y[2]))

# Concatenate tensors into one tensor
training_data = TensorDataset(training_data_x, input_mask_train, training_labels_y)
test_data = TensorDataset(test_data_x, input_mask_test, test_labels_y)

# MODEL TRAINING
print('-----------------------')

# split observations into batches, train for 2 epochs
batch_size = 64
num_train_epochs = 1

train_sampler = RandomSampler(training_data)

# Dataloader is used to iterate over batches
train_dataloader = DataLoader(training_data,
                              sampler=train_sampler,
                              batch_size=batch_size)

# //: divide and discard remainder (Division ohne Rest)
t_total = len(train_dataloader) // num_train_epochs

# this is used in the huggingface example
# num_training_steps = train_epochs * len(train_dataloader)
# print(num_training_steps)

# Learning variables
#print(len(training_data))
#print(num_train_epochs)
#print(batch_size)
#print(t_total)

# set learning parameters
learning_rate = 5e-5
adam_epsilon = 1e-8
warmup_steps = 100

# for parameter adjustment
optimizer = AdamW(model.parameters(), lr=learning_rate, eps=adam_epsilon)

#  define a learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=warmup_steps,
                                            num_training_steps=t_total)

# if available, the gpu will be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# output is too long
# print(model)

# tqdm is for tracking the training progress
progress_bar = tqdm(range(t_total))

# Put model in 'train'
model.train()

print("\nTraining Model.......Start Time " + str(time.strftime('%c')))

# wrap epoch and batches in loops
for epoch in range(num_train_epochs):
    print('\nStart of epoch %d' % (epoch,))
    # iterate over data/batches
    for step, batch in enumerate(train_dataloader):
        # all gradients reset at start of every iteration
        # model.zero_grad()

        # print(device)
        batch = tuple(t.to(device) for t in batch)

        # set inputs of the model
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2]}

        # forward to model
        outputs = model(**inputs)

        # deviation (loss)
        # loss = outputs[0]
        loss = outputs.loss
        print("\r%f" % loss, end='')

        # Backpropagation
        loss.backward()

        # limit gradients to 1.0 to prevent exploding gradients -->  is deprecated
        # torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)

        # update parameters and learning rate
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

print('--- End Time ' + str(time.strftime('%c')))

# save model
# model.save_pretrained('BERT_model_1')

# EVALUATE MODEL

test_batch_size = 64

# this time samples are fed sequentially, not random
test_sampler = SequentialSampler(test_data)

# dataloader for the test data
test_dataloader = DataLoader(test_data,
                             sampler=test_sampler,
                             batch_size=test_batch_size)

# load model if needed, by
# model = model.from_pretrained('BERT_model_1')

# Init for prediction and labels
preds = None
out_labels_ids = None

# evaluation mode
model.eval()

for batch in tqdm(test_dataloader, desc="Evaluating"):

    model.to(device)
    batch = tuple(t.to(device) for t in batch)

    # no gradients tracking because testing
    with torch.no_grad():
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2]}


        # ** -> converts method parameter from dictionary to keyword arguments bsp: model(input_ids=batch[0], attention_mask= batch[0], usw)
        outputs = model(**inputs)

        # get loss
        _tmp_eval_loss, logits = outputs[:2]

        # batch items check
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids,
                                      inputs['labels'].detach().cpu().numpy(),
                                      axis=0)

print("--------------------------------------------------")

#print(preds.argmax(axis=1)) #[ 0.52554715 -0.07667161], [0.19761527  0.06251662]
#print(type(preds)) # <class 'numpy.ndarray'>

#print(out_label_ids)# [0 1 0 1 0 0 0 0 0 1 0
#print(type(out_label_ids)) #<class 'numpy.ndarray'>

#print(preds.shape)
#print(out_label_ids.shape)

# argmax returns index of max value of the two values in each array
preds = preds.argmax(axis=1)

# preds is now 1-dim
#print(preds.shape)

acc_score = accuracy_score(preds, out_label_ids)
print('\nAccuracy Score on Test data ', acc_score)

#print("accuracy andere methode: ")
#acc = metrics.accuracy_score(test_labels_y, preds)

# Get prediction and accuracy
preds = preds
actual =out_label_ids

print("------------ output metrics calculation...---------")
#f1_score = metrics.f1_score(test_labels_y, preds)
#recall = metrics.recall_score(test_labels_y, preds)
#precision = metrics.precision_score(test_labels_y, preds)
#print(acc_score, f1_score, recall, precision)


print(classification_report(out_label_ids, preds))


# True pos = (1,1), True neg = (0,0), False pos = (1,0), False neg = (0,1)
TP = np.count_nonzero(preds * actual)
TN = np.count_nonzero((preds - 1) * (actual - 1))
FP = np.count_nonzero(preds * (actual - 1))
FN = np.count_nonzero((preds - 1) * actual)

print("True Positives", TP)
print("True Negatives", TN)
print("False Positives", FP)
print("False Negatives", FN)

# function for confusion matrix
def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

# plot confusionmatrix
plot_cm(actual, preds)

print(type(X_train))
print(X_train[:3])
