import pandas as pd

def read_tabular(fpath, fargs):
    suffix = fpath.split('.')[-1]
    if suffix == 'csv':
        return pd.read_csv(fpath, **fargs)
    elif suffix in ['xlsx', 'xlsm']:
        return pd.read_excel(fpath, **fargs, engine='openpyxl')
    elif suffix == 'xls':
        return pd.read_excel(fpath, **fargs)
    raise NotImplementedError(f"Unsupported file type: {fpath}")
