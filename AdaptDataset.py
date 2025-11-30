import pandas as pd

def concat_datasets(dfs):
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def split_validation_train(dfs, test_ratio=0.3):
    total_dfs = len(dfs)
    validation_size = int(total_dfs * test_ratio)
    train_size = total_dfs - validation_size

    train_dfs = dfs[:train_size]
    validation_dfs = dfs[train_size:]
    train_df  = concat_datasets(train_dfs)
    validation_df = concat_datasets(validation_dfs)

    return train_df, validation_df