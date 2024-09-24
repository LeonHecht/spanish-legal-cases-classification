
from datasets import Dataset
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


class LossDifferenceCallback(TrainerCallback):
    def __init__(self, loss_diff_threshold):
        # Threshold for difference in loss
        self.loss_diff_threshold = loss_diff_threshold
        self.training_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Store training loss from each logging step
        if logs is not None:
            if 'loss' in logs:
                self.training_losses.append(logs['loss'])

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        # Calculate the average training loss
        average_training_loss = sum(self.training_losses) / len(self.training_losses) if self.training_losses else float('inf')
        # Get the validation loss from the evaluation metrics
        validation_loss = metrics.get("eval_loss", float("inf"))
        # print the average training loss and validation loss with two decimal places
        print(f"\n\nAverage training loss: {average_training_loss:.2f}, Validation loss: {validation_loss:.2f}\n\n")

        # Calculate the difference and decide if training should stop
        loss_diff = abs((validation_loss - average_training_loss) / average_training_loss)
        if loss_diff > self.loss_diff_threshold:
            print(f"Stopping training due to loss difference: {loss_diff}")
            control.should_training_stop = True

        # Reset training losses after evaluation
        self.training_losses = []


class CustomDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            # First, try to access attribute from the DataParallel itself
            return super().__getattr__(name)
        except AttributeError:
            # If failed, try to access it from the wrapped model
            return getattr(self.module, name)


def create_datasets(tokenizer, train_texts, val_texts, test_texts, y_train, y_val, y_test=None, max_length=256):
    # Tokenize and encode the text data
    def tokenize_data(texts, labels=None):
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All elements in 'texts' must be strings.")
        encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        if labels is not None:
            encodings["labels"] = labels
        return encodings

    train_encodings = tokenize_data(train_texts.to_list(), y_train.to_list())
    val_encodings = tokenize_data(val_texts.to_list(), y_val.to_list())
    test_encodings = tokenize_data(test_texts.to_list(), y_test.to_list() if y_test is not None else None)

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
    # weighted_model.load_state_dict(model.state_dict())  # Copy the weights from the original model
    # Wrap the model with DataParallel to use multiple GPUs
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs!")
    #     model = CustomDataParallel(model)
    # model.cuda()  # Ensure the model is on the correct device
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
    remove_unused_columns=False,  # Keep all columns
    greater_is_better=True  # Higher f1 score is better
    )
    return training_args


def get_trainer(model, training_args, train_dataset, val_dataset, hyperparameters):
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=hyperparameters.get("early_stopping_patience", 4))]
    # callbacks=[LossDifferenceCallback(loss_diff_threshold=loss_diff_threshold)]
    )
    return trainer


def predict(trainer, test_dataset):
    predictions = trainer.predict(test_dataset)
    return predictions


def get_labels(predictions):
    test_pred_labels = np.argmax(predictions.predictions, axis=-1)
    print("Predicted Labels", test_pred_labels)
    return test_pred_labels

def run(model_checkpoint, num_labels,
        train_texts, val_texts, test_texts,
        y_train, y_val, y_test=None,
        hyperparameters=not None):
    
    max_length = hyperparameters.get("max_length", 256)
    # load BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
    train_dataset, val_dataset, test_dataset = create_datasets(tokenizer, train_texts, val_texts, test_texts, y_train, y_val, y_test=None, max_length=max_length)
    classes = np.unique(y_train)
    print("Type of classes:", type(classes))
    print("Classes:", classes)
    model = load_model(model_checkpoint, num_labels, classes, y_train, hyperparameters['use_weighted_loss'])
    # model = torch.nn.DataParallel(model)
    training_args = training_arguments(hyperparameters)
    trainer = get_trainer(model, training_args, train_dataset, val_dataset, hyperparameters)
    # Train the model
    trainer.train()
    predictions = predict(trainer, test_dataset)
    test_pred_labels = get_labels(predictions)
    # Generate and print the classification report
    if num_labels == 2:
        print(classification_report(y_test, test_pred_labels, target_names=['Class 0', 'Class 1']))
    elif num_labels == 3:
        print(classification_report(y_test, test_pred_labels, target_names=['Class 1', 'Class 2', 'Class 3']))
    else:
        print(classification_report(y_test, test_pred_labels, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3']))
    return test_pred_labels


def run_optimization(model_checkpoint, num_labels, train_texts, val_texts, test_texts, y_train, y_val, y_test):
    import optuna
    # num_train_epochs = 3

    def objective(trial):
        # Define the hyperparameters to be optimized
        learning_rate = trial.suggest_float("learning_rate", 5e-7, 5e-5, log=True)
        # num_train_epochs = trial.suggest_int("num_train_epochs", 5, 50, log=True)
        # batch_size = trial.suggest_int("batch_size", 16, 48, step=16)
        batch_size = 64
        weight_decay = trial.suggest_float("weight_decay", 0.0, 0.05)
        # loss_diff_threshold = trial.suggest_float("loss_diff_threshold", 0.1, 0.5, step=0.1)

        # Update the training arguments with the suggested hyperparameters
        training_args = training_arguments(epochs=100, batch_size=batch_size, weight_decay=weight_decay, learning_rate=learning_rate)
        trainer = get_trainer(model, training_args, train_dataset, val_dataset, loss_diff_threshold=None)

        # Train the model and get the evaluation results
        trainer.train()
        eval_result = trainer.evaluate()

        # Return the metric to be maximized/minimized
        return eval_result["eval_f1"]

    classes = np.unique(y_train)
    model = load_model(model_checkpoint, num_labels, classes, y_train)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    train_dataset, val_dataset, test_dataset = create_datasets(tokenizer, train_texts, val_texts, test_texts, y_train, y_val, y_test)

    # Run the optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=150)

    # Print the best hyperparameters
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Train the model with the best hyperparameters
    best_training_args = training_arguments(epochs=100, batch_size=64, weight_decay=trial.params["weight_decay"], learning_rate=trial.params["learning_rate"])

    trainer = get_trainer(model, best_training_args, train_dataset, val_dataset, loss_diff_threshold=None)

    # Train the model
    trainer.train()

    # Predict the test dataset
    predictions = predict(trainer, test_dataset)

    # Generate and print the classification report
    test_pred_labels = get_labels(predictions)
    print(classification_report(y_test, test_pred_labels, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3']))

    import os
    # print if path "../data/trial_summaries" exists
    print("Trial summary path exists: ", os.path.exists("data/trial_summaries"))

    # Save the trial summary to a CSV file
    sum_df = study.trials_dataframe()
    sum_df.to_csv(f'data/trial_summaries/summary_{model_checkpoint}_{datetime.now()}.csv', index=False)
    print("Trial summary:\n", sum_df)

    return test_pred_labels


def run_lora(model_checkpoint, train_texts, val_texts, test_texts, y_train, y_val, y_test):
    from peft import LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    train_dataset, val_dataset, test_dataset = create_datasets(tokenizer, train_texts, val_texts, test_texts, y_train, y_val, y_test)
    model = load_model(model_checkpoint)
    print(model)

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
    lora_config = LoraConfig(
        target_modules = [
            f"distilbert.transformer.layer.{i}.attention.q_lin" for i in range(6)] + [
            f"distilbert.transformer.layer.{i}.attention.k_lin" for i in range(6)] + [
            f"distilbert.transformer.layer.{i}.attention.v_lin" for i in range(6)] + [
            f"distilbert.transformer.layer.{i}.attention.out_lin" for i in range(6)] + [
            f"distilbert.transformer.layer.{i}.ffn.lin1" for i in range(6)] + [
            f"distilbert.transformer.layer.{i}.ffn.lin2" for i in range(6)
        ],
        r=64,
        task_type="SEQ_CLS",
        lora_alpha=128,
        lora_dropout=0.05,       # 0.05
        use_rslora=True
    )

    # load LoRA model
    lora_model = get_peft_model(model, lora_config)

    training_args = training_arguments()
    trainer = get_trainer(lora_model, training_args, train_dataset, val_dataset)
    
    # Train the model
    print("Training the model...")
    print("Verifying train dataset structure...")
    print(train_dataset[0])  # This should print the first element to check structure
    trainer.train()

    predictions = predict(trainer, test_dataset)
    test_pred_labels = get_labels(predictions)
    # Generate and print the classification report
    print(classification_report(y_test, test_pred_labels, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3']))
    return test_pred_labels


from sklearn.model_selection import ParameterGrid
from datetime import datetime

def run_grid_search(model_checkpoint, num_labels, train_texts, val_texts, test_texts, y_train, y_val, y_test):
    # Define the grid of hyperparameters to search
    param_grid = {
        'learning_rate': [1e-6, 3e-6, 5e-6],
        'warmup_steps': [1000, 1500, 2500]
    }

    # Initialize variables to store the best results
    best_f1 = 0
    best_params = {}

    # Create the datasets
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
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
