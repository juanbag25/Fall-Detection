import pandas as pd
import random

def concat_datasets(dfs):
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def split_validation_train(dfs, test_ratio=0.3,seed=None):

    if seed is not None:
        random.seed(seed)
    dfs_shuffled = dfs.copy()
    random.shuffle(dfs_shuffled)

    # Calcular tamaños
    total_dfs = len(dfs_shuffled)
    validation_size = int(total_dfs * test_ratio)

    # Dividir
    validation_dfs = dfs_shuffled[:validation_size]
    train_dfs = dfs_shuffled[validation_size:]

    # Concatenar
    train_df = concat_datasets(train_dfs)
    validation_df = concat_datasets(validation_dfs)

    return train_df, validation_df
