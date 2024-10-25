hyperparameters1 = {
    'epochs': 5,
    'batch_size': 8,
    'weight_decay': 0.01,
    'learning_rate': 5e-6,      # lower learning rate
    'warmup_steps': 1000,
    'metric_for_best_model': "f1",
    'early_stopping_patience': 4,
    'max_length': 1024,
    'stride': 512,
    'use_weighted_loss': False,
    'max_grad_norm': 0,
    'gradient_accumulation_steps': 1,
    'mixed_precision': True,
    'read_datasets': True,
    'datasets_read_path': 'datasets/google2',
    'save_datasets': False,
    'datasets_save_path': None,
}

hyperparameters2 = {
    'epochs': 10,
    'batch_size': 8,
    'weight_decay': 0.01,
    'learning_rate': 5e-6,      # lower learning rate
    'warmup_steps': 1000,
    'metric_for_best_model': "f1",
    'early_stopping_patience': 4,
    'max_length': 1024,
    'stride': 512,
    'use_weighted_loss': False,
    'max_grad_norm': 0,
    'gradient_accumulation_steps': 1,
    'mixed_precision': True,
    'read_datasets': True,
    'datasets_read_path': 'datasets/google2',
    'save_datasets': False,
    'datasets_save_path': None,
}

hyperparameters3 = {
    'epochs': 5,
    'batch_size': 4,
    'weight_decay': 0.01,
    'learning_rate': 2e-5,
    'warmup_steps': 1000,
    'metric_for_best_model': "f1",
    'early_stopping_patience': 4,
    'max_length': 1024,
    'stride': 512,
    'use_weighted_loss': False,
    'max_grad_norm': 0,
    'gradient_accumulation_steps': 1,
    'mixed_precision': False,
    'read_datasets': True,
    'datasets_read_path': 'datasets/google2',
    'save_datasets': False,
    'datasets_save_path': None,
}

hyperparameters4 = {
    'epochs': 5,
    'batch_size': 4,
    'weight_decay': 0.01,
    'learning_rate': 2e-5,
    'warmup_steps': 1000,
    'metric_for_best_model': "f1",
    'early_stopping_patience': 4,
    'max_length': 1024,
    'stride': 512,
    'use_weighted_loss': True,
    'max_grad_norm': 0,
    'gradient_accumulation_steps': 1,
    'mixed_precision': False,
    'read_datasets': True,
    'datasets_read_path': 'datasets/google2',
    'save_datasets': False,
    'datasets_save_path': None,
}

# model_checkpoint = 'mrm8488/longformer-base-4096-spanish-finetuned-squad'
# model_checkpoint = 'state-spaces/mamba2-130m'
model_checkpoint = 'Narrativa/legal-longformer-base-4096-spanish'

# corpus_path='corpus/cleaned_corpus_google_sin_resuelve.csv'
# corpus_path='corpus/corpus.csv'
# corpus_path4='corpus/corpus_google_min_line_len4_min_par_len2.csv'
corpus_path2='corpus/corpus_google_min_line_len2_min_par_len2.csv'

experiments = [
    {
        'name': 'mixed precision (28)',
        'hyperparameters': hyperparameters1,
        'model_checkpoint': model_checkpoint,
        'corpus_path': corpus_path2
    },
    {
        'name': 'mixed precision with 10 epochs (29)',
        'hyperparameters': hyperparameters2,
        'model_checkpoint': model_checkpoint,
        'corpus_path': corpus_path2
    },
    {
        'name': 'repeating experiment of line 24 (30)',
        'hyperparameters': hyperparameters3,
        'model_checkpoint': model_checkpoint,
        'corpus_path': corpus_path2
    },
    {
        'name': 'repeating experiment of line 24 but with weighted loss (31)',
        'hyperparameters': hyperparameters4,
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
                        mamba=False)
