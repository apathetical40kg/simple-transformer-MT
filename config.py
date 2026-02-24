from pathlib import Path

def get_config():
    return {
        'batch_size': 16,
        'num_epochs': 20,
        'lr': 1e-4,
        'seq_len': 350,
        'embedding_dim': 512,
        'lang_src': 'de',
        'lang_tgt': 'en',
        'model_folder': 'weights',
        'model_basename': 'tmodel_',
        'preload': '01',
        'tokenizer_file': 'tokenizer_{0}.json',
        'experiment_name': 'runs/tmodel'
    }

def get_weigths_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f'{model_basename}{epoch}.pt'
    return str(Path('.') / model_folder / model_filename)
