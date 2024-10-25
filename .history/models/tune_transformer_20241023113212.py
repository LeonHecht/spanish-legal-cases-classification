
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, TrainerControl, TrainerState
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import classification_report
# import library for timestamp
from datetime import datetime
from transformers import EarlyStoppingCallback
import torch
from transformers import TrainerCallback

from sklearn.utils.class_weight import compute_class_weight

print("---------------------------------------------")
print("---------------------------------------------")
print("Number of GPUs:", torch.cuda.device_count())
print("---------------------------------------------")
print("---------------------------------------------")


import torch.nn as nn
import torch.nn.functional as F


class WeightedSmoothCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights, smoothing=0.1):
        super(WeightedSmoothCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        n_classes = inputs.size(1)
        assert len(self.class_weights) == n_classes, "Class weights should match number of classes"

        # Create the smoothed label
        targets = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.smoothing) * targets + self.smoothing / n_classes
        
        # Apply class weights
        if self.class_weights is not None:
            class_weights = torch.tensor(self.class_weights, dtype=inputs.dtype, device=inputs.device)
            targets = targets * class_weights.unsqueeze(0)
        
        # Calculate weighted smooth loss
        loss = F.log_softmax(inputs, dim=1).mul(targets).sum(dim=1).mean() * -1
        return loss


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight  # class weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


class WeightedAutoModel(AutoModelForSequenceClassification):
    def __init__(self, config, class_weights):
        super().__init__(config)
        self.class_weights = class_weights
        self.focal_loss = FocalLoss(weight=self.class_weights, gamma=2.0)
        # self.criterion = FocalLoss(alpha=0.25, gamma=2.0)

    def compute_loss_cross_ent(self, model_output, labels):
        # model_output: tuple of (logits, ...)
        logits = model_output[0]
        # Assuming using CrossEntropyLoss, adjust accordingly if using a different loss
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        return loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    
    def compute_loss(self, model_output, labels):
        logits = model_output[0]
        loss_fct = WeightedSmoothCrossEntropyLoss(class_weights=self.class_weights)
        smoothed_loss = loss_fct(logits, labels)
        cross_ent_loss = self.compute_loss_cross_ent(model_output, labels)
        focal_loss = self.focal_loss(logits, labels)
        return (smoothed_loss + cross_ent_loss + focal_loss)/3

    # def compute_loss_combined(self, model_output, labels):
    #     cross_ent_lost = self.compute_loss_cross_ent(model_output, labels)
    #     logits = model_output[0]
    #     focal_loss = self.focal_loss(logits, labels)
    #     return (cross_ent_lost + focal_loss)/2

    # def compute_loss_weighed_focal(self, model_output, labels):
    #     # model_output: tuple of (logits, ...)
    #     logits = model_output[0]
    #     # Assuming using FocalLoss, adjust accordingly if using a different loss
    #     loss_fct = FocalLoss()
    #     return loss_fct(logits.view(-1, self.num_labels), labels.view(-1))


def create_datasets(tokenizer, train_texts, val_texts, test_texts, y_train, y_val, y_test=None, max_length=256, stride=None):
    # Tokenize and encode the text data
    def tokenize_data(texts, labels=None):
        """ Tokenize data without sliding window """
        print("Using normal tokenization without sliding window")
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All elements in 'texts' must be strings.")
        encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        if labels is not None:
            encodings["labels"] = labels
        return encodings

    def tokenize_data_sliding_window(texts, labels=None):
        """ Tokenize data and implement sliding window with stride """
        print(f"Using sliding window with stride {stride}")

        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All elements in 'texts' must be strings.")

        encodings = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "docID": []
        }

        total_chunks = 0

        for i, text in enumerate(texts):
            text_length = len(tokenizer(text)["input_ids"])
            # print(f"Original text length for text {i}: {text_length}")

            # Tokenize text with sliding window
            tokenized_example = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                stride=stride,  # Implementing sliding window with stride
                return_overflowing_tokens=True,  # To get multiple windows per text
                return_offsets_mapping=False  # Not needed for this use case
            )

            # print("tokenized_example:", tokenized_example)
            
            num_chunks = len(tokenized_example['input_ids'])  # Number of chunks created
            # print(f"Number of chunks for text {i}: {num_chunks}")
            total_chunks += num_chunks

            # Extend the encodings with each chunk and corresponding label
            encodings["input_ids"].extend(tokenized_example["input_ids"])
            encodings["attention_mask"].extend(tokenized_example["attention_mask"])

            # Assign the same document ID to all chunks of the same text
            encodings["docID"].extend([i] * num_chunks)
            
            if labels is not None:
                # Assign the same label to all chunks of the original text
                encodings["labels"].extend([labels[i]] * num_chunks)

        print(f"Total number of chunks: {total_chunks}")

        return encodings

    print("Stride:", stride)
    
    # Tokenize the data
    if stride is None:
        train_encodings = tokenize_data(train_texts.to_list(), y_train.to_list())
        val_encodings = tokenize_data(val_texts.to_list(), y_val.to_list())
        test_encodings = tokenize_data(test_texts.to_list(), y_test.to_list() if y_test is not None else None)
    else:
        print("Tokenizing train data...")
        train_encodings = tokenize_data_sliding_window(train_texts.to_list(), y_train.to_list())
        print("Tokenizing val data...")
        val_encodings = tokenize_data_sliding_window(val_texts.to_list(), y_val.to_list())
        print("Tokenizing test data...")
        test_encodings = tokenize_data_sliding_window(test_texts.to_list(), y_test.to_list() if y_test is not None else None)

    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_dict(train_encodings)
    val_dataset = Dataset.from_dict(val_encodings)
    test_dataset = Dataset.from_dict(test_encodings)
    
    print("Sample train input_ids:", train_dataset['input_ids'][0])

    return train_dataset, val_dataset, test_dataset        


def load_model(model_checkpoint, num_labels, classes, y_train, is_weighted):
    if is_weighted:
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
        model = WeightedAutoModel.from_pretrained(model_checkpoint, num_labels=num_labels)
        model.class_weights = class_weights_tensor
        print("using weighted model")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
        print("using automodel")
    return model


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)
    metrics = {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}
    return metrics


def training_arguments(hyperparameters):
    epochs = hyperparameters.get("epochs", 100)
    batch_size = hyperparameters.get("batch_size", 16)
    weight_decay = hyperparameters.get("weight_decay", 0.01)
    learning_rate = hyperparameters.get("learning_rate", 5e-6)
    warmup_steps = hyperparameters.get("warmup_steps", 1000)
    metric = hyperparameters.get("metric_for_best_model", "f1")
    max_grad_norm = hyperparameters.get("max_grad_norm", 0)
    gradient_accumulation_steps = hyperparameters.get("gradient_accumulation_steps", 1)

    print("Training arguments")
    print("Batch size:", batch_size)
    print("Weight decay:", weight_decay)
    print("Learning rate:", learning_rate)
    print("Warmup steps:", warmup_steps)
    print("Metric for best model:", metric)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,  # limit the number of saved checkpoints
        load_best_model_at_end=True,
        metric_for_best_model=metric,
        remove_unused_columns=True,
        greater_is_better=True,  # Higher f1 score is better
        max_grad_norm=max_grad_norm,
        gradient_accumulation_steps=gradient_accumulation_steps
    )
    return training_args


def get_trainer(model, training_args, tokenizer, train_dataset, val_dataset, hyperparameters):
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=hyperparameters.get("early_stopping_patience", 4))]
    )
    return trainer


def predict(trainer, test_dataset):
    predictions = trainer.predict(test_dataset)
    test_pred_labels = np.argmax(predictions.predictions, axis=-1)
    print("Predicted Labels", test_pred_labels)
    return test_pred_labels


def predict_with_avg_logits(trainer, test_dataset, tokenizer, max_length, stride):
    # Run predictions with the trainer
    raw_predictions = trainer.predict(test_dataset)
    
    # Extracting the predictions (logits) and labels
    predictions = raw_predictions.predictions  # These are the logits
    labels = raw_predictions.label_ids

    print("INSIDE PREDICT WITH AVG LOGITS")
    print("Predictions", predictions)
    print("Labels", labels)
    
     # Create a dictionary to store the logits for each docID
    doc_logits = {}
    
    # Iterate over the dataset to group the logits by docID
    for i in range(len(test_dataset)):
        doc_id = test_dataset[i]["docID"]
        
        # Check if the doc_id is already in the dictionary
        if doc_id not in doc_logits:
            doc_logits[doc_id] = []

        # Add the logits for the current chunk
        doc_logits[doc_id].append(predictions[i])
    
    # Now, average the logits for each document
    averaged_logits = []
    for doc_id, logits in doc_logits.items():
        avg_logits = np.mean(logits, axis=0)  # Average logits across chunks
        averaged_logits.append(avg_logits)

    # Convert averaged logits to final predictions
    final_predictions = np.argmax(averaged_logits, axis=-1)  # Apply softmax if needed before

    print("Predicted Labels", final_predictions)

    return final_predictions  # Return final document-level predictions


def run(model_checkpoint, num_labels,
        train_texts, val_texts, test_texts,
        y_train, y_val, y_test=None,
        hyperparameters=not None):
    
    max_length = hyperparameters.get("max_length", 256)
    print("Max length:", max_length)
    print("Stride", hyperparameters["stride"])
    # load BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
    
    read_datasets = hyperparameters.get("read_datasets", False)
    datasets_read_path = hyperparameters.get("datasets_read_path", "datasets")

    if read_datasets:
        # read datasets
        print("Reading google datasets from disk")
        train_dataset = load_from_disk("datasets/train_dataset")
        val_dataset = load_from_disk("datasets/val_dataset")
        test_dataset = load_from_disk("datasets/test_dataset")
    else:
        train_dataset, val_dataset, test_dataset = create_datasets(tokenizer,
                                                               train_texts, val_texts, test_texts,
                                                               y_train, y_val, y_test,
                                                               max_length=max_length, stride=hyperparameters["stride"])


    # save datasets
    train_dataset.save_to_disk("datasets/train_dataset_google")
    val_dataset.save_to_disk("datasets/val_dataset_google")
    test_dataset.save_to_disk("datasets/test_dataset_google")
    print("Datasets saved to disk")

    classes = np.unique(y_train)
    print("Type of classes:", type(classes))
    print("Classes:", classes)
    model = load_model(model_checkpoint, num_labels, classes, y_train, hyperparameters['use_weighted_loss'])
    # model = torch.nn.DataParallel(model)
    training_args = training_arguments(hyperparameters)
    trainer = get_trainer(model, training_args, tokenizer, train_dataset, val_dataset, hyperparameters)
    # Train the model
    trainer.train()
    if hyperparameters["stride"] is not None:
        predictions = predict_with_avg_logits(trainer, test_dataset, tokenizer, max_length, hyperparameters["stride"])
    else:
        predictions = predict(trainer, test_dataset)
    # Generate and print the classification report
    if num_labels == 2:
        print(classification_report(y_test, predictions, target_names=['Class 0', 'Class 1']))
    elif num_labels == 3:
        print(classification_report(y_test, predictions, target_names=['Class 1', 'Class 2', 'Class 3']))
    else:
        print(classification_report(y_test, predictions, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3']))
    return predictions


def run_lora(model_checkpoint, num_labels,
             train_texts, val_texts, test_texts,
             y_train, y_val, y_test,
             hyperparameters):
    from peft import LoraConfig, get_peft_model, TaskType

    max_length = hyperparameters.get("max_length", 256)

    # tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
    train_dataset, val_dataset, test_dataset = create_datasets(tokenizer,
                                                               train_texts, val_texts, test_texts,
                                                               y_train, y_val, y_test,
                                                               max_length, stride=hyperparameters["stride"])
    
    # train_dataset = load_from_disk("datasets/train_dataset")
    # val_dataset = load_from_disk("datasets/val_dataset")
    # test_dataset = load_from_disk("datasets/test_dataset")

    classes = np.unique(y_train)
    print("Type of classes:", type(classes))
    print("Classes:", classes)
    model = load_model(model_checkpoint, num_labels, classes, y_train, hyperparameters['use_weighted_loss'])

    # print(model)

    # LoRA config
    # target_modules=["q_proj", "v_proj"],
        # target_modules = [
        #     layer_name
        #     for i in range(48)  # Adjust based on the number of layers in your model
        #     for layer_name in [
        #         f"deberta.encoder.layer.{i}.attention.self.query_proj",
        #         f"deberta.encoder.layer.{i}.attention.self.key_proj",
        #         f"deberta.encoder.layer.{i}.attention.self.value_proj",
        #         f"deberta.encoder.layer.{i}.attention.output.dense"
        #     ]
        # ],
    # lora_config = LoraConfig(
    #     target_modules = [
    #         f"distilbert.transformer.layer.{i}.attention.q_lin" for i in range(6)] + [
    #         f"distilbert.transformer.layer.{i}.attention.k_lin" for i in range(6)] + [
    #         f"distilbert.transformer.layer.{i}.attention.v_lin" for i in range(6)] + [
    #         f"distilbert.transformer.layer.{i}.attention.out_lin" for i in range(6)] + [
    #         f"distilbert.transformer.layer.{i}.ffn.lin1" for i in range(6)] + [
    #         f"distilbert.transformer.layer.{i}.ffn.lin2" for i in range(6)
    #     ],
    lora_config = LoraConfig(
        # target_modules=[
        #     f"roberta.encoder.layer.{i}.attention.self.query" for i in range(12)] + [
        #     f"roberta.encoder.layer.{i}.attention.self.key" for i in range(12)] + [
        #     f"roberta.encoder.layer.{i}.attention.self.value" for i in range(12)] + [
        #     f"roberta.encoder.layer.{i}.attention.output.dense" for i in range(12)] + [
        #     f"roberta.encoder.layer.{i}.intermediate.dense" for i in range(12)] + [
        #     f"roberta.encoder.layer.{i}.output.dense" for i in range(12)
        # ],
        r=8,
        # task_type="SEQ_CLS",
        task_type=TaskType.SEQ_CLS,
        lora_alpha=32,
        lora_dropout=0.05,       # 0.05
        use_rslora=True
    )

    # load LoRA model
    model = get_peft_model(model, lora_config)

    # Example usage with your LoRA model
    model.print_trainable_parameters()

    print("YES DATA PARALLEL")
    model = torch.nn.DataParallel(model)

    # # Print out the parameters and whether they require gradient updates
    # for name, param in model.named_parameters():
    #     print(f"{name}: requires_grad={param.requires_grad}")

    training_args = training_arguments(hyperparameters)
    trainer = get_trainer(model, training_args, tokenizer, train_dataset, val_dataset, hyperparameters)

    print(torch.cuda.memory_summary())

    # Train the model
    print("Training the model...")
    print("Verifying train dataset structure...")
    print(train_dataset[0])  # This should print the first element to check structure
    trainer.train()

    print(torch.cuda.memory_summary())

    if hyperparameters["stride"] is not None:
        predictions = predict_with_avg_logits(trainer, test_dataset, tokenizer, max_length, hyperparameters["stride"])
    else:
        predictions = predict(trainer, test_dataset)
    # Generate and print the classification report
    if num_labels == 2:
        print(classification_report(y_test, predictions, target_names=['Class 0', 'Class 1']))
    elif num_labels == 3:
        print(classification_report(y_test, predictions, target_names=['Class 1', 'Class 2', 'Class 3']))
    else:
        print(classification_report(y_test, predictions, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3']))
    return predictions

from sklearn.model_selection import ParameterGrid
from datetime import datetime

def run_grid_search(model_checkpoint, num_labels,
                    train_texts, val_texts, test_texts,
                    y_train, y_val, y_test,
                    hyperparameters):
    # Define the grid of hyperparameters to search
    param_grid = {
        'learning_rate': [2e-6, 2e-5, 2e-4],
        # 'warmup_steps': [1000, 1500, 2500]
    }

    # Initialize variables to store the best results
    best_f1 = 0
    best_params = {}

    max_length = hyperparameters.get("max_length", 256)

    # Create the datasets
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
    train_dataset, val_dataset, test_dataset = create_datasets(tokenizer, train_texts, val_texts, test_texts, y_train, y_val, y_test)

    # Load the initial model
    classes = np.unique(y_train)
    model = load_model(model_checkpoint, num_labels, classes, y_train)

    trial = 1
    # Iterate over all combinations of parameters
    for params in ParameterGrid(param_grid):
        print(f"\nTrial {trial}/{len(ParameterGrid(param_grid))}\n")
        print("Current Parameters:\n", params)

        training_args = training_arguments(
            epochs=100,
            batch_size=16,
            weight_decay=0.01,
            learning_rate=params['learning_rate'],
            warmup_steps=params['warmup_steps']
        )
        trainer = get_trainer(model, training_args, train_dataset, val_dataset, loss_diff_threshold=None)

        # Train the model
        trainer.train()
        eval_result = trainer.evaluate()

        # Check if the current trial's F1-score is better
        if eval_result["eval_f1"] > best_f1:
            best_f1 = eval_result["eval_f1"]
            best_params = params

    print("Best Parameters Found:")
    for key, value in best_params.items():
        print(f"    {key}: {value}")

    # Train the model with the best parameters
    final_training_args = training_arguments(
        epochs=100,
        batch_size=best_params['batch_size'],
        weight_decay=best_params['weight_decay'],
        learning_rate=best_params['learning_rate'],
        warmup_steps=best_params['warmup_steps']
    )
    final_trainer = get_trainer(model, final_training_args, train_dataset, val_dataset, loss_diff_threshold=None)
    final_trainer.train()

    # Predict the test dataset
    predictions = predict(final_trainer, test_dataset)
    test_pred_labels = get_labels(predictions)
    print(classification_report(y_test, test_pred_labels, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3']))

    # Optionally save the results summary to a CSV
    results_summary_path = f'data/trial_summaries/summary_{model_checkpoint}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'
    print("Saving trial summary to:", results_summary_path)
    with open(results_summary_path, 'w') as file:
        file.write(f"Best F1-Score: {best_f1}\n")
        file.write("Best Parameters:\n")
        for param, value in best_params.items():
            file.write(f"{param}: {value}\n")

    return test_pred_labels