import torch
from torch.utils.data import Dataset
import torch.nn as nn
from sklearn.metrics import f1_score
import torch.nn.functional as F
import torch.optim as optim

criterion = nn.CrossEntropyLoss()

# 1. Define Dataset Class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
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
    

# 6. Evaluation function
def evaluate_model(model, dataloader, epoch, device, phase='val'):
    if phase == 'test':
        print("\nEvaluating on test data...")

    model.eval()
    correct_predictions = 0
    total_predictions = 0
    total_loss = 0

    all_labels = []
    all_predictions = []

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

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted_class.cpu().numpy())
            
    accuracy = correct_predictions / total_predictions
    f1 = f1_score(all_labels, all_predictions, average='macro')
    average_loss = total_loss / len(dataloader)
    # # Log validation loss, accuracy, and f1 score
    # writer.add_scalar(f'{phase}/Loss', average_loss, epoch)
    # writer.add_scalar(f'{phase}/Accuracy', accuracy, epoch)
    # writer.add_scalar(f'{phase}/F1_Score', f1, epoch)

    print(f"{phase.capitalize()} Accuracy: {accuracy * 100:.2f}% | F1 Score: {f1:.4f} | Loss: {average_loss:.4f}")
    if phase == 'test':
        # print predictions
        print("Predictions:", all_predictions)


# 5. Training loop
def train_model(model, train_dataloader, val_dataloader, learning_rate, epochs, device):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.train()
        loss_sum = 0
        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask').to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            # Log loss
            loss_sum += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Print batch progress
            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f"Batch {batch_idx}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

            # writer.add_scalar('Loss/train', loss.item(), epoch * len(train_dataloader) + batch_idx)

        print(f"Train Loss: {loss_sum/len(train_dataloader):.4f}")
        
        # calculate validation accuracy after each epoch
        evaluate_model(model, val_dataloader, epoch, device, phase='val')


