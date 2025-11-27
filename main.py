# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter
import AdaptDataset as adapt

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

