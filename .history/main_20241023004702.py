from utils import print_time
import pandas as pd
from sklearn.model_selection import train_test_split
from models import tune_transformer


# # replace original test labels with predicted labels
# df_test['label'] = test_pred_labels

# # save the dataframe with predicted labels to a csv file
# print("Saving predictions to csv...")
# df_test.to_csv('corpus/prediction_task3.tsv', sep='\t', index=False)


# ## Mamba

from transformers import MambaForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# import Loading Bar
from peft import LoraConfig, get_peft_model, TaskType

from utils.train import MambaForTextClassification, TextDataset, train_model, evaluate_model


# Hyperparameters
epochs = 1
batch_size = 16
learning_rate = 2e-5
max_length = 512

# Create a SummaryWriter to log metrics
# writer = SummaryWriter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 3. Initialize model, tokenizer, and dataset
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
mamba_model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
classification_model = MambaForTextClassification(mamba_model, num_labels=2)

print(classification_model)

lora_config = LoraConfig(
        target_modules=[
            "mamba_model.backbone.layers.*.mixer.in_proj",
            "mamba_model.backbone.layers.*.mixer.x_proj",
            "mamba_model.backbone.layers.*.mixer.dt_proj",
            "mamba_model.backbone.layers.*.mixer.out_proj"
        ],
        r=8,
        # task_type="SEQ_CLS",
        task_type=TaskType.SEQ_CLS,
        lora_alpha=32,
        lora_dropout=0.05,       # 0.05
        use_rslora=True
    )

classification_model = get_peft_model(classification_model, lora_config)
classification_model.print_trainable_parameters()

classification_model = nn.DataParallel(classification_model)
classification_model.to(device)

freeze_mamba = False
if freeze_mamba:
    for param in classification_model.module.mamba_model.parameters():
        param.requires_grad = False

# Tokenize and create dataset
train_dataset = TextDataset(train_texts.tolist(), y_train.tolist(), tokenizer, max_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TextDataset(val_texts.tolist(), y_val.tolist(), tokenizer, max_length)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TextDataset(test_texts.tolist(), y_test.tolist(), tokenizer, max_length)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Train the model
train_model(classification_model, train_dataloader, val_dataloader, learning_rate, epochs, device)

# Evaluate the model on test data
evaluate_model(classification_model, test_dataloader, epoch=-1, device=device, phase='test')

# # Close TensorBoard writer
# writer.close()

# ## Print End Time
print_time.print_("End-Time")

def print_df_info(df):
    print(df.head())
    print(df['text'][0])
    print(df.describe())

def print_split_info(train_texts, val_texts, test_texts, y_train):
    print('Train samples:', train_texts.shape[0])
    print('Validation samples:', val_texts.shape[0])
    print('Test samples:', test_texts.shape[0])
    print()
    # print labels distribution in train
    print(y_train.value_counts())

def save_texts(train_texts, val_texts, test_texts):
    print("Converting train, val and test texts to csv...")
    train_texts.to_csv('corpus/train_texts.csv', index=False, header=False)
    val_texts.to_csv('corpus/val_texts.csv', index=False, header=False)
    test_texts.to_csv('corpus/test_texts.csv', index=False, header=False)

def run_standard_finetuning(model_checkpoint,
                            train_texts, val_texts, test_texts,
                            y_train, y_val, y_test,
                            hyperparameters):
    test_pred_labels = tune_transformer.run(model_checkpoint, 2,
                                            train_texts, val_texts, test_texts,
                                            y_train, y_val, y_test,
                                            hyperparameters=hyperparameters)


def main(hyperparameters, model_checkpoint, corpus_path, read_datasets):
    print_time.print_("Start-Time")

    df = pd.read_csv(corpus_path, usecols=['text', 'label'])

    print_df_info(df)

    train_texts, temp_texts, y_train, y_temp = train_test_split(
    df['text'], df['label'],
    test_size=0.3, random_state=42
    )

    val_texts, test_texts, y_val, y_test = train_test_split(
        temp_texts, y_temp,
        test_size=0.5, random_state=42
    )

    print_split_info(train_texts, val_texts, test_texts, y_train)

    print("------------------------------------")
    print("Model:", model_checkpoint)
    print("------------------------------------")

    run_standard_finetuning(model_checkpoint,
                            train_texts, val_texts, test_texts,
                            y_train, y_val, y_test,
                            hyperparameters)