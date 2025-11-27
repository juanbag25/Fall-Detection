import pandas as pd

def concat_datasets(dfs):
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def split_test_train(dfs, test_ratio=0.3):
    total_dfs = len(dfs)
    test_size = int(total_dfs * test_ratio)
    train_size = total_dfs - test_size

    train_dfs = dfs[:train_size]
    test_dfs = dfs[train_size:]

    train_df  = concat_datasets(train_dfs)
    test_df = concat_datasets(test_dfs)

    return train_df, test_df
