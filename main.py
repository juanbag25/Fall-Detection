# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter
import AdaptDataset as adapt
import SlidingWindow
import metrics
from IPython.display import display

filepath = "exp1.csv"
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "juanbautistagitba/caminata-y-caida-datos-imu-mpu6050-en-espalda",
  filepath
)

print("First 5 records:", df.head())

dfs = []
for i in range (1,8):
  filepath= f"exp{i}.csv" 
  df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "juanbautistagitba/caminata-y-caida-datos-imu-mpu6050-en-espalda",
    filepath
  )
  dfs.append(df)

train_df, test_df = adapt.split_test_train(dfs, test_ratio=0.3)

print("Train dataset size:", len(train_df))
print("Test dataset size:", len(test_df))

train_windows = SlidingWindow.SlidingWindows(
    train_df,
    window_size=8,
    step_size=2,
    metrics_functions=[
        metrics.get_mean_acc_module,
        metrics.get_std_acc_module,
        metrics.get_max_acc_module,
        metrics.get_min_acc_module,
        metrics.get_acc_module_diff,
        metrics.get_mean_jerk_acc_module,
        metrics.get_kurtosis_acc_module,
        metrics.get_energy_acc_module,
        metrics.get_total_zero_crossings,
        metrics.get_zero_crossings_accx,
        metrics.get_zero_crossings_accy,
        metrics.get_zero_crossings_accz
    ]
)


#print(train_windows.create_features_dataframe())
print(train_windows.features_df.head())

