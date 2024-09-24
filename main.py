#!/usr/bin/env python
# coding: utf-8

# ## Print Start time

# In[1]:


from utils import print_time

print_time.print_("Start-Time")


# ## Specify modes

# 1. Baseline
# 2. Combined Loss Function
# 3. Max_length = 256
# 4. Paraphrase Augmentation
# 5. Traditional Augmentation
# 6. Everything Combined
# 7. Everything Combined (Full texts) (?)

# In[2]:


# modes_all = [{'deploying': False,       # 1. Baseline
#          'undersampling': False,
#          'paraphrase_aug': False,
#          'traditional_aug': False},
         
#          {'deploying': False,       # 2. Combined Loss Function
#          'undersampling': False,
#          'paraphrase_aug': False,
#          'traditional_aug': False},
         
#          {'deploying': False,       # 3. Max_length=256
#          'undersampling': False,
#          'paraphrase_aug': False,
#          'traditional_aug': False},         
         
#          {'deploying': False,       # 4. Paraphrase Aug
#          'undersampling': False,
#          'paraphrase_aug': True,
#          'traditional_aug': False},         
         
#          {'deploying': False,       # 5. Traditional Aug
#          'undersampling': False,
#          'paraphrase_aug': False,
#          'traditional_aug': True},         
         
#          {'deploying': False,       # 6. Everything combined
#          'undersampling': False,
#          'paraphrase_aug': True,
#          'traditional_aug': True},         
#         ]


# ## Hyperparameters

# In[3]:


# Constants
epochs = 15
batch_size = 16
weight_decay = 0.01
learning_rate = 5e-5
warmup_steps = 1000
metric_for_best_model = "f1"
early_stopping_patience = 6
max_length = 700

hyperparameters = {
    'epochs': epochs,     # 1. Baseline
    'batch_size': batch_size,
    'weight_decay': weight_decay,
    'learning_rate': learning_rate,
    'warmup_steps': warmup_steps,
    'metric_for_best_model': metric_for_best_model,
    'early_stopping_patience': early_stopping_patience,
    'max_length': max_length,
    'use_weighted_loss': False
    }


# In[4]:


# hyperparameters_all = [
#     {'epochs': epochs,     # 1. Baseline
#     'batch_size': batch_size,
#     'weight_decay': weight_decay,
#     'learning_rate': learning_rate,
#     'warmup_steps': warmup_steps,
#     'metric_for_best_model': metric_for_best_model,
#     'early_stopping_patience': early_stopping_patience,
#     'max_length': 128,
#     'use_weighted_loss': False
#     },
    
#     {'epochs': epochs,     # 2. Combined Loss Function
#     'batch_size': batch_size,
#     'weight_decay': weight_decay,
#     'learning_rate': learning_rate,
#     'warmup_steps': warmup_steps,
#     'metric_for_best_model': metric_for_best_model,
#     'early_stopping_patience': early_stopping_patience,
#     'max_length': 128,
#     'use_weighted_loss': True
#     },

#     {'epochs': epochs,     # 3. Max_length=256
#     'batch_size': batch_size,
#     'weight_decay': weight_decay,
#     'learning_rate': learning_rate,
#     'warmup_steps': warmup_steps,
#     'metric_for_best_model': metric_for_best_model,
#     'early_stopping_patience': early_stopping_patience,
#     'max_length': 256,
#     'use_weighted_loss': True
#     },
    
#     {'epochs': epochs,     # 4. Paraphrase Aug
#     'batch_size': batch_size,
#     'weight_decay': weight_decay,
#     'learning_rate': learning_rate,
#     'warmup_steps': warmup_steps,
#     'metric_for_best_model': metric_for_best_model,
#     'early_stopping_patience': early_stopping_patience,
#     'max_length': 128,
#     'use_weighted_loss': True
#     },

#     {'epochs': epochs,     # 5. Traditional Aug
#     'batch_size': batch_size,
#     'weight_decay': weight_decay,
#     'learning_rate': learning_rate,
#     'warmup_steps': warmup_steps,
#     'metric_for_best_model': metric_for_best_model,
#     'early_stopping_patience': early_stopping_patience,
#     'max_length': 128,
#     'use_weighted_loss': True
#     },
    
#     {'epochs': epochs,     # 6. Everything combined
#     'batch_size': batch_size,
#     'weight_decay': weight_decay,
#     'learning_rate': learning_rate,
#     'warmup_steps': warmup_steps,
#     'metric_for_best_model': metric_for_best_model,
#     'early_stopping_patience': early_stopping_patience,
#     'max_length': 256,
#     'use_weighted_loss': True
#     },

# ]


# ## Specify Model

# In[5]:


# model_checkpoint = 'mrm8488/longformer-base-4096-spanish-finetuned-squad'
model_checkpoint = 'state-spaces/mamba2-130m'
# model_checkpoint = 'Narrativa/legal-longformer-base-4096-spanish'
# model_checkpoint = 'distilbert-base-uncased'
# model_checkpoint = 'roberta-base'
# model_checkpoint = 'bert-large-uncased'
# model_checkpoint = 'xlnet-base-cased'
# model_checkpoint = 'xlnet-large-cased'
# model_checkpoint = 'xlm-roberta-large'
# model_checkpoint = 'microsoft/deberta-v2-xxlarge'


# ## Load df

# In[6]:


import pandas as pd

corpus_path='corpus/corpus_final_corregido.txt'
df = pd.read_csv(corpus_path, sep='\t', usecols=['Contenido Txt', 'Resultado binario de la acción'])

# rename columns
df.rename(columns = {'Contenido Txt':'text', 'Resultado binario de la acción':'label'}, inplace = True)


# In[7]:


print(df['text'][1])
print(df['label'][1])


# In[8]:


# shuffle df
df = df.sample(frac=1).reset_index(drop=True)

# cut df to 500 rows
df = df[:500]


# In[9]:


print(df.head())


# ## Split data

# In[10]:


from sklearn.model_selection import train_test_split

train_texts, temp_texts, y_train, y_temp = train_test_split(
    df['text'], df['label'],
    test_size=0.3, random_state=42
)

val_texts, test_texts, y_val, y_test = train_test_split(
    temp_texts, y_temp,
    test_size=0.5, random_state=42
)


# In[11]:


print('Train samples:', train_texts.shape[0])
print('Validation samples:', val_texts.shape[0])
print('Test samples:', test_texts.shape[0])
print()

# print labels distribution in train
print(y_train.value_counts())


# ## Get train_df

# In[12]:


# import pandas as pd
# train_df = pd.DataFrame({'text': train_texts, 'label': y_train})

# # Contar el número de publicaciones en cada categoría
# class_counts = train_df['label'].value_counts()
# print("Class distribution before augmenting with paraphrased texts:\n", class_counts)


# ## Augment train_df by augmented texts

# In[13]:


# import os

# if deploying:
#     paraphrase_path = "data/augmented_dfs_trainval/"
#     aug_path = "data/traditional_augmentation_trainval/"
# else:
#     paraphrase_path = "data/augmented_dfs_train/"
#     aug_path = "data/traditional_augmentation_train/"
    
# if paraphrase_aug:
#     paraphrased_1_df_1 = pd.read_csv(paraphrase_path + 'Paraphrase1/paraphrased_class_1.csv', usecols=['text', 'label', 'keyword'])
#     paraphrased_1_df_2 = pd.read_csv(paraphrase_path + 'Paraphrase2/paraphrased_class_1.csv', usecols=['text', 'label', 'keyword'])
#     paraphrased_1_df_3 = pd.read_csv(paraphrase_path + 'Paraphrase3/paraphrased_class_1.csv', usecols=['text', 'label', 'keyword'])
    
#     paraphrased_2_df_1 = pd.read_csv(paraphrase_path + 'Paraphrase1/paraphrased_class_2.csv', usecols=['text', 'label', 'keyword'])

#     paraphrased_3_df_1 = pd.read_csv(paraphrase_path + 'Paraphrase1/paraphrased_class_3.csv', usecols=['text', 'label', 'keyword'])
#     paraphrased_3_df_2 = pd.read_csv(paraphrase_path + 'Paraphrase2/paraphrased_class_3.csv', usecols=['text', 'label', 'keyword'])
#     paraphrased_3_df_3 = pd.read_csv(paraphrase_path + 'Paraphrase3/paraphrased_class_3.csv', usecols=['text', 'label', 'keyword'])
#     paraphrased_3_df_4 = pd.read_csv(paraphrase_path + 'Paraphrase4/paraphrased_class_3.csv', usecols=['text', 'label', 'keyword'])

#     paraphrased_df = pd.concat([paraphrased_1_df_1, paraphrased_1_df_2, paraphrased_1_df_3, paraphrased_2_df_1, paraphrased_3_df_1, paraphrased_3_df_2, paraphrased_3_df_3, paraphrased_3_df_4], ignore_index=True)

#     # Add keywords to paraphrased dfs
#     paraphrased_df = preprocessing.add_keywords(paraphrased_df, model_checkpoint)

#     train_df = pd.concat([train_df, paraphrased_df], ignore_index=True)
    
# if traditional_aug:
#     punct_df = pd.read_csv(aug_path + 'punct_df.csv', usecols=['text', 'label', 'keyword'])
#     # punct_df = punct_df.loc[punct_df['label'] != 0]
#     # punct_df = punct_df.loc[punct_df['label'] != 2]

#     rnd_del_df = pd.read_csv(aug_path + 'rnd_del_df.csv', usecols=['text', 'label', 'keyword'])
#     # rnd_del_df = rnd_del_df.loc[rnd_del_df['label'] != 0]
#     # rnd_del_df = rnd_del_df.loc[rnd_del_df['label'] != 2]

#     rnd_swap_df = pd.read_csv(aug_path + 'rnd_swap_df.csv', usecols=['text', 'label', 'keyword'])
#     # rnd_swap_df = rnd_swap_df.loc[rnd_swap_df['label'] != 0]
#     # rnd_swap_df = rnd_swap_df.loc[rnd_swap_df['label'] != 2]

#     rnd_insert_df = pd.read_csv(aug_path + 'rnd_insert_df.csv', usecols=['text', 'label', 'keyword'])
#     # rnd_insert_df = rnd_insert_df.loc[rnd_insert_df['label'] != 0]
#     # rnd_insert_df = rnd_insert_df.loc[rnd_insert_df['label'] != 2]

#     aug_df = pd.concat([punct_df, rnd_del_df, rnd_swap_df, rnd_insert_df], ignore_index=True)

#     # Add keywords to augmented dfs
#     aug_df = preprocessing.add_keywords(aug_df, model_checkpoint)

#     # merge df with paraphrased dfs
#     # train_df = pd.concat([train_df, paraphrased_df, aug_df], ignore_index=True)
#     train_df = pd.concat([train_df, aug_df], ignore_index=True)
        


# ## Cut Classes to X texts

# In[14]:


# if undersampling:
#     # Size of each class after sampling (Hyperparameter)
#     # Class 0 has 796 samples and was not augmented
#     class_size = 1500

#     # Sample 200 texts from each class (or as many as are available for classes with fewer than 200 examples)
#     sampled_dfs = []
#     for label in train_df['label'].unique():
#         class_sample_size = min(len(train_df[train_df['label'] == label]), class_size)
#         sampled_dfs.append(train_df[train_df['label'] == label].sample(n=class_sample_size, random_state=42))

#     # Concatenate the samples to create a balanced training DataFrame
#     train_df = pd.concat(sampled_dfs, ignore_index=True)


# ## Extract texts and labels from train_df

# In[15]:


# shuffled_train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
# # Now you can extract the texts and labels
# train_texts = shuffled_train_df['text']
# print("Train texts balanced", train_texts)
# # print datatype of y train values
# y_train = shuffled_train_df['label']
# print("Datatype of y_train", type(y_train))
# print("y_train balanced", y_train)


# ## Print train_df class distribution after cutting/before augmentation

# In[16]:


# # Contar el número de publicaciones en cada categoría
# class_counts = train_df['label'].value_counts()
# print("Class distribution after cutting:\n", class_counts)


# In[17]:


# import matplotlib.pyplot as plt

# df_plot = train_df.copy()

# label_mapping = {1: 'positive', 2: 'neutral', 3: 'negative', 0: 'unrelated'}
# df_plot['label'] = df_plot['label'].map(label_mapping)

# # Contar el número de publicaciones en cada categoría
# class_counts = df_plot['label'].value_counts()
# print(class_counts)

# # Crear un gráfico de barras
# plt.figure(figsize=(8, 6))
# class_counts.plot(kind='bar')
# plt.title('Distribución de clases')
# plt.xlabel('Clase')
# plt.ylabel('Número de publicaciones')
# plt.xticks(rotation=0)
# plt.show()


# ## Run Model

# In[18]:


print("Converting train, val and test texts to csv...")
train_texts.to_csv('corpus/train_texts.csv', index=False, header=False)
val_texts.to_csv('corpus/val_texts.csv', index=False, header=False)
test_texts.to_csv('corpus/test_texts.csv', index=False, header=False)


# In[19]:


# from models import tune_transformer
# import torch
# from transformers import BertTokenizer
# from torch.optim import AdamW
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.metrics import accuracy_score
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, AutoModel
# import numpy as np

# print("------------------------------------")
# print("Model:", model_checkpoint)
# print("------------------------------------")

# # # Hyperparameters
# # batch_size = 16
# # epochs = 3
# # learning_rate = 5e-5

# # # Load pre-trained BERT and tokenizer
# # tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
# # model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2, trust_remote_code=True)

# # # Tokenize the texts
# # def tokenize_texts(texts, tokenizer):
# #     return tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=4096)

# # # Convert Pandas Series to list of strings
# # train_texts = train_texts.tolist()
# # val_texts = val_texts.tolist()
# # test_texts = test_texts.tolist()

# # y_train = y_train.tolist()
# # y_val = y_val.tolist()
# # y_test = y_test.tolist()

# # # Tokenize train, validation, and test datasets
# # train_encodings = tokenize_texts(train_texts, tokenizer)
# # val_encodings = tokenize_texts(val_texts, tokenizer)
# # test_encodings = tokenize_texts(test_texts, tokenizer)

# # # Convert labels to tensor format
# # y_train_tensor = torch.tensor(y_train)
# # y_val_tensor = torch.tensor(y_val)
# # y_test_tensor = torch.tensor(y_test)

# # # Create DataLoader for train and validation sets
# # train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], y_train_tensor)
# # val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], y_val_tensor)
# # test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], y_test_tensor)

# # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# # # Optimizer
# # optimizer = AdamW(model.parameters(), lr=learning_rate)

# # # Fine-tuning the model
# # device = torch.device('cuda:0')
# # model.to(device)

# # def train(model, train_loader, val_loader, optimizer, epochs=3):
# #     model.train()
# #     for epoch in range(epochs):
# #         total_train_loss = 0
# #         for batch in train_loader:
# #             optimizer.zero_grad()

# #             input_ids, attention_mask, labels = [x.to(device) for x in batch]

# #             outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
# #             loss = outputs.loss
# #             loss.backward()
# #             optimizer.step()

# #             total_train_loss += loss.item()

# #         avg_train_loss = total_train_loss / len(train_loader)
# #         print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}")

# #         # Validation after each epoch
# #         val_acc = evaluate(model, val_loader)
# #         print(f"Validation Accuracy after epoch {epoch + 1}: {val_acc:.4f}")

# # def evaluate(model, loader):
# #     model.eval()
# #     predictions, true_labels = [], []
# #     with torch.no_grad():
# #         for batch in loader:
# #             input_ids, attention_mask, labels = [x.to(device) for x in batch]

# #             outputs = model(input_ids=input_ids, attention_mask=attention_mask)
# #             logits = outputs.logits

# #             predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
# #             true_labels.extend(labels.cpu().numpy())

# #     acc = accuracy_score(true_labels, predictions)
# #     return acc

# # # Training the model
# # train(model, train_loader, val_loader, optimizer, epochs)

# # # Evaluating the model on the test set
# # test_acc = evaluate(model, test_loader)
# # print(f"Test Accuracy: {test_acc:.4f}")

# from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba2-130m-hf")
# model = MambaForCausalLM.from_pretrained("state-spaces/mamba2-130m-hf")
# input_ids = tokenizer("Hey how are you doing?", return_tensors="pt")["input_ids"]

# out = model.generate(input_ids, max_new_tokens=10)
# print(tokenizer.batch_decode(out))

# # test_pred_labels = tune_transformer.run(model_checkpoint, 2,
# #                                         train_texts, val_texts, test_texts,
# #                                         y_train, y_val, y_test,
# #                                         hyperparameters=hyperparameters)

# # # replace original test labels with predicted labels
# # df_test['label'] = test_pred_labels

# # # save the dataframe with predicted labels to a csv file
# # print("Saving predictions to csv...")
# # df_test.to_csv('corpus/prediction_task3.tsv', sep='\t', index=False)


# ## Mamba

# In[20]:


from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
# import f1_score from sklearn
from sklearn.metrics import f1_score
# import Loading Bar
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Define Dataset Class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}  # Remove batch dimension
        inputs["labels"] = torch.tensor(label, dtype=torch.long)
        return inputs

# 2. Modify the model to add a classification head
class MambaForTextClassification(nn.Module):
    def __init__(self, model, num_labels):
        super(MambaForTextClassification, self).__init__()
        self.mamba_model = model
        self.classifier = nn.Linear(self.mamba_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        # Get hidden states from the language model
        outputs = self.mamba_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # Get the last hidden state

        # Pool the hidden states (take the hidden state corresponding to [CLS] token or mean pooling)
        pooled_output = hidden_states[:, 0, :]  # Using the first token's embedding (usually [CLS] token)

        # Pass the pooled output through the classifier
        logits = self.classifier(pooled_output)
        return logits

# 3. Initialize model, tokenizer, and dataset
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
mamba_model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
classification_model = MambaForTextClassification(mamba_model, num_labels=2)

classification_model.to(device)

batch_size = 16
freeze_mamba = True

if freeze_mamba:
    for param in classification_model.mamba_model.parameters():
        param.requires_grad = False

# Tokenize and create dataset
train_dataset = TextDataset(train_texts.tolist(), y_train.tolist(), tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TextDataset(val_texts.tolist(), y_val.tolist(), tokenizer)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TextDataset(test_texts.tolist(), y_test.tolist(), tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# 4. Define optimizer and loss function
optimizer = optim.AdamW(classification_model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# 6. Evaluation function
def evaluate_model(model, dataloader, test=False):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    total_loss = 0

    progress_bar_eval = tqdm(dataloader, desc="Evaluating", leave=False)    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask').to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask=attention_mask)
            probs = F.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1)
            
            correct_predictions += (predicted_class == labels).sum().item()
            total_predictions += labels.size(0)

            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            # increase Loading Bar
            progress_bar_eval.update(1)
            progress_bar_eval.set_postfix({"loss": loss.item()})
    
    accuracy = correct_predictions / total_predictions
    # calculate f1 score
    f1 = f1_score(labels.cpu().numpy(), predicted_class.cpu().numpy(), average='macro')
    average_loss = total_loss / len(dataloader)
    if test:
        print(f"Test Accuracy: {accuracy * 100:.2f}% | F1 Score: {f1:.4f} | Loss: {average_loss:.4f}")
    else:
        print(f"Validation Accuracy: {accuracy * 100:.2f}% | F1 Score: {f1:.4f} | Loss: {average_loss:.4f}")

# 5. Training loop
def train_model(model, dataloader, optimizer, criterion, epochs=3):
    model.train()
    for epoch in range(epochs):
        print("\nEpoch", epoch + 1)
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch +1}", leave=False)
        # reset progress bar
        progress_bar.reset()
        loss_sum = 0
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask').to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss_sum += loss.item()
            
            # Backpropagation
            loss.backward()
            optimizer.step()

            # increase Loading Bar
            progress_bar.update(1)
            progress_bar.set_postfix({"Loss": loss.item()})
            
        print(f"Train Loss: {loss_sum/len(dataloader):.4f}")
        # calculate validation accuracy after each epoch
        evaluate_model(model, val_dataloader)

# Train the model
train_model(classification_model, train_dataloader, optimizer, criterion, epochs=3)

# Evaluate the model (using the same data for demonstration purposes)
evaluate_model(classification_model, test_dataloader)


# ## Print End Time

# In[21]:


print_time.print_("End-Time")

