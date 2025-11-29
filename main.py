# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter
import AdaptDataset as adapt
import SlidingWindow
import metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import feature_analysis as fa


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
#print(train_windows.features_df.head())

test_windows = SlidingWindow.SlidingWindows(
    test_df,
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

#print(test_windows.create_features_dataframe())
#print(test_windows.features_df.head())

features=train_windows.features_df.drop(columns=['secayo'])
scaler=StandardScaler()
scaled_features=scaler.fit_transform(features)

df_scaled = pd.DataFrame(scaled_features, columns=features.columns)

print("Scaled features head:")
print(df_scaled.head())


fa.correlation_analysis(df_scaled)

##Vemos que hay una correlación alta entre mean_acc_module y energy_acc_module
##Vemos que hay una correlación alta entre std_acc_module y acc_module_diff
#Vemos que hay una correlación alta entre max_acc_module y std_acc_module
#En el futuro probar eliminar algunas de estas variables para ver si mejora el rendimiento del modelo


fa.pca_analysis(df_scaled)

#Podemos ver que las dos primeras componentes principales explican casi 60% de la varianza
#La varianza  se explica con 9 componentes
#Un 98%de la varianza se explica con 8 componentes