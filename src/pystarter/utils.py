import os, random, yaml, numpy as np, torch, pandas as pd
import pystarter.definitions as D

def read_tabular(fpath, fargs):
    suffix = fpath.split('.')[-1]
    if suffix == 'csv':
        return pd.read_csv(fpath, **fargs)
    elif suffix in ['xlsx', 'xlsm']:
        return pd.read_excel(fpath, **fargs, engine='openpyxl')
    elif suffix == 'xls':
        return pd.read_excel(fpath, **fargs)
    raise NotImplementedError(f"Unsupported file type: {fpath}")

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def load_config(configf):
    with open(os.path.join(D.CONFIGS, configf), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config