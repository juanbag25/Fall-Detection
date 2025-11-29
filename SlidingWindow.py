import metrics as metrics
import pandas as pd

class SlidingWindows:
    def __init__(self, df,window_size, step_size, metrics_functions=[]):
        self.df = df
        self.window_size = window_size
        self.step_size = step_size
        self.windows = self.sliding_window()
        self.metrics_functions = metrics_functions
        self.features_df = self.create_features_dataframe()

    def sliding_window(self):
        self.windows = []
        for start in range(0, len(self.df) - self.window_size + 1, self.step_size):
            end = start + self.window_size
            window = self.df.iloc[start:end]
            self.windows.append(window)
        return self.windows

    
    def calculate_metrics(self):
        all_metrics = []
        for window in self.windows:
            window_metrics = {}
            for func in self.metrics_functions:
                metric_name = func.__name__
                metric_value = func(window)
                window_metrics[metric_name] = metric_value
            window_label = window['secayo'].mode()[0]
            window_metrics['secayo'] = window_label
            all_metrics.append(window_metrics)
        return all_metrics


    def create_features_dataframe (self):
        metrics_list = self.calculate_metrics()
        self.features_df = pd.DataFrame(metrics_list)
        return self.features_df
    

    

    

