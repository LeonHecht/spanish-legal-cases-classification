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
    'epochs': epochs,
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

# model_checkpoint = 'mrm8488/longformer-base-4096-spanish-finetuned-squad'
# model_checkpoint = 'state-spaces/mamba2-130m'
model_checkpoint = 'Narrativa/legal-longformer-base-4096-spanish'

# corpus_path='corpus/cleaned_corpus_google_sin_resuelve.csv'
# corpus_path='corpus/corpus.csv'
corpus_path4='corpus/corpus_google_min_line_len4_min_par_len2.csv'
corpus_path2='corpus/corpus_google_min_line_len2_min_par_len2.csv'

experiments = [
    {
        'name': 'excel sheet line 23',
        'hyperparameters': hyperparameters,
        'model_checkpoint': model_checkpoint,
        'corpus_path': corpus_path4
    },
    {
        'name': 'excel sheet line 24',
        'hyperparameters': hyperparameters,
        'model_checkpoint': model_checkpoint,
        'corpus_path': corpus_path2
    },
]

import main

for experiment in experiments:
    print("---------------------------------------------")
    print(f"\n\nRunning experiment {experiment['name']}")
    print(f"Hyperparameters: {experiment['hyperparameters']}")
    print(f"Model checkpoint: {experiment['model_checkpoint']}")
    print(f"Corpus path: {experiment['corpus_path']}\n\n")
    print("---------------------------------------------")

    main.run_experiment(hyperparameters=experiment['hyperparameters'],
                        model_checkpoint=experiment['model_checkpoint'],
                        corpus_path=experiment['corpus_path'],
                        mamba=False,
                        read_datasets=False)
