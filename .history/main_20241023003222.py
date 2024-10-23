#!/usr/bin/env python
# coding: utf-8

# ## Print Start time

# In[1]:


from utils import print_time

print_time.print_("Start-Time")


# ## Hyperparameters

# In[2]:


# Constants
epochs = 5
batch_size = 4
weight_decay = 0.01
learning_rate = 2e-5
warmup_steps = 1000
metric_for_best_model = "f1"
early_stopping_patience = 4
max_length = 1024
stride = 512

hyperparameters = {
    'epochs': epochs,     # 1. Baseline
    'batch_size': batch_size,
    'weight_decay': weight_decay,
    'learning_rate': learning_rate,
    'warmup_steps': warmup_steps,
    'metric_for_best_model': metric_for_best_model,
    'early_stopping_patience': early_stopping_patience,
    'max_length': max_length,
    'stride': stride,
    'use_weighted_loss': False
    }


# ## Specify Model

# In[3]:


# model_checkpoint = 'mrm8488/longformer-base-4096-spanish-finetuned-squad'
# model_checkpoint = 'state-spaces/mamba2-130m'
model_checkpoint = 'Narrativa/legal-longformer-base-4096-spanish'
# model_checkpoint = 'distilbert-base-uncased'
# model_checkpoint = 'roberta-base'
# model_checkpoint = 'bert-large-uncased'
# model_checkpoint = 'xlnet-base-cased'
# model_checkpoint = 'xlnet-large-cased'
# model_checkpoint = 'xlm-roberta-large'
# model_checkpoint = 'microsoft/deberta-v2-xxlarge'


# ## Load df

# In[4]:


import pandas as pd

# corpus_path='corpus/cleaned_corpus_google_sin_resuelve.csv'
# corpus_path='corpus/corpus.csv'
corpus_path='corpus/corpus_google_min_line_len4_min_par_len2.csv'
# df = pd.read_csv(corpus_path, sep='\t', usecols=['Contenido Txt', 'Resultado binario de la acción'])
df = pd.read_csv(corpus_path, usecols=['text', 'label'])

# rename columns
# df.rename(columns = {'Contenido Txt':'text', 'Resultado binario de la acción':'label'}, inplace = True)


# In[5]:


# # Separate the entries with label 1
# df_label_1 = df[df['label'] == 1]

# # Randomly sample the same number of entries from label 0
# df_label_0 = df[df['label'] == 0].sample(n=len(df_label_1), random_state=42)

# # Combine both balanced subsets
# df = pd.concat([df_label_1, df_label_0])

# # Shuffle the combined DataFrame to mix label 0 and 1
# df = df.sample(frac=1, random_state=42)


# In[6]:


# cut df to X rows
# df = df[:100]


# In[7]:


print(df.head())


# In[8]:


print(df['text'][0])


# In[ ]:


df.describe()


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


# ## Run Model

# In[12]:


print("Converting train, val and test texts to csv...")
train_texts.to_csv('corpus/train_texts.csv', index=False, header=False)
val_texts.to_csv('corpus/val_texts.csv', index=False, header=False)
test_texts.to_csv('corpus/test_texts.csv', index=False, header=False)


# In[13]:


from models import tune_transformer

print("------------------------------------")
print("Model:", model_checkpoint)
print("------------------------------------")

test_pred_labels = tune_transformer.run(model_checkpoint, 2,
                                        train_texts, val_texts, test_texts,
                                        y_train, y_val, y_test,
                                        hyperparameters=hyperparameters)

# # replace original test labels with predicted labels
# df_test['label'] = test_pred_labels

# # save the dataframe with predicted labels to a csv file
# print("Saving predictions to csv...")
# df_test.to_csv('corpus/prediction_task3.tsv', sep='\t', index=False)


# ## Mamba

# In[15]:


# from transformers import MambaForCausalLM, AutoTokenizer
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# # import Loading Bar
# from peft import LoraConfig, get_peft_model, TaskType

# from utils.train import MambaForTextClassification, TextDataset, train_model, evaluate_model


# # Hyperparameters
# epochs = 1
# batch_size = 16
# learning_rate = 2e-5
# max_length = 512

# # Create a SummaryWriter to log metrics
# # writer = SummaryWriter()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # 3. Initialize model, tokenizer, and dataset
# tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
# mamba_model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
# classification_model = MambaForTextClassification(mamba_model, num_labels=2)

# print(classification_model)

# lora_config = LoraConfig(
#         target_modules=[
#             "mamba_model.backbone.layers.*.mixer.in_proj",
#             "mamba_model.backbone.layers.*.mixer.x_proj",
#             "mamba_model.backbone.layers.*.mixer.dt_proj",
#             "mamba_model.backbone.layers.*.mixer.out_proj"
#         ],
#         r=8,
#         # task_type="SEQ_CLS",
#         task_type=TaskType.SEQ_CLS,
#         lora_alpha=32,
#         lora_dropout=0.05,       # 0.05
#         use_rslora=True
#     )

# classification_model = get_peft_model(classification_model, lora_config)
# classification_model.print_trainable_parameters()

# classification_model = nn.DataParallel(classification_model)
# classification_model.to(device)

# freeze_mamba = False
# if freeze_mamba:
#     for param in classification_model.module.mamba_model.parameters():
#         param.requires_grad = False

# # Tokenize and create dataset
# train_dataset = TextDataset(train_texts.tolist(), y_train.tolist(), tokenizer, max_length)
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# val_dataset = TextDataset(val_texts.tolist(), y_val.tolist(), tokenizer, max_length)
# val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# test_dataset = TextDataset(test_texts.tolist(), y_test.tolist(), tokenizer, max_length)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# # Train the model
# train_model(classification_model, train_dataloader, val_dataloader, learning_rate, epochs, device)

# # Evaluate the model on test data
# evaluate_model(classification_model, test_dataloader, epoch=-1, device=device, phase='test')

# # # Close TensorBoard writer
# # writer.close()


# ## Print End Time

# In[21]:


print_time.print_("End-Time")

