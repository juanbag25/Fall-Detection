import metrics as metrics

class SlidingWindows:
    def __init__(self, window_size, step_size):
        self.window_size = window_size
        self.step_size = step_size
        self.windows = []

    def sliding_window(self, df):
        self.windows = []
        for start in range(0, len(df) - self.window_size + 1, self.step_size):
            end = start + self.window_size
            window = df.iloc[start:end]
            self.windows.append(window)
        return self.windows
    
    def calculate_metrics(self, metrics_functions):
        all_metrics = [] 
        for window in self.windows:
            window_metrics = {}
            for func in metrics_functions:
                metric_name = func.__name__
                window_metrics[metric_name] = func(window)
            all_metrics.append(window_metrics)

    

    

