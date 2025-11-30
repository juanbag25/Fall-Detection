# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter
import AdaptDataset as adapt
import SlidingWindow
import metrics
import pandas as pd
import feature_analysis as fa

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

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

train_df, validation_df = adapt.split_validation_train(dfs, test_ratio=0.3)

print("Train dataset size:", len(train_df))
print("Validation dataset size:", len(validation_df))

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

validation_windows = SlidingWindow.SlidingWindows(
    validation_df,
    window_size=5,
    step_size=1,
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

features=train_windows.features_df.drop(columns=['secayo'])

scaler=StandardScaler()
scaled_features=scaler.fit_transform(features)
df_scaled = pd.DataFrame(scaled_features, columns=features.columns)
print("Scaled features head:")
print(df_scaled.head())

# fa.correlation_analysis(df_scaled)

#Vemos que hay una correlación alta entre mean_acc_module y energy_acc_module
#Vemos que hay una correlación alta entre std_acc_module y acc_module_diff
#Vemos que hay una correlación alta entre max_acc_module y std_acc_module
#En el futuro probar eliminar algunas de estas variables para ver si mejora el rendimiento del modelo

#!!! fa.pca_analysis(df_scaled)

#Podemos ver que las dos primeras componentes principales explican casi 60% de la varianza
#La varianza  se explica con 8 componentes
#Un 98%de la varianza se explica con 7 componentes. Tenemos como 4 componentes que no aportan más información

features2=features.drop(columns=['get_acc_module_diff','get_max_acc_module','get_energy_acc_module'])
scaled_features2=scaler.fit_transform(features2)
df_scaled2 = pd.DataFrame(scaled_features2, columns=features2.columns)

#!!! fa.correlation_analysis(df_scaled2)

# Ahora observamos que las caracteristicas con mas correlación entre sí son zero crossings_accx y zero_crossings_accy con zero_crossings_total con 75% de correlación
#Saquemos zero crossings_total.

features3=features2.drop(columns=['get_total_zero_crossings'])
scaled_features3=scaler.fit_transform(features3)
df_scaled3 = pd.DataFrame(scaled_features3, columns=features3.columns)

#!!!fa.correlation_analysis(df_scaled3)

# Ahora no hay correlaciones altas entre las variables

#!!!fa.pca_analysis(df_scaled3)

##Pasamos a los modelos


X_train = df_scaled3
Y_train = train_windows.features_df['secayo']
X_validation = validation_windows.features_df.drop(columns=['secayo','get_total_zero_crossings','get_acc_module_diff','get_max_acc_module','get_energy_acc_module'])
Y_validation = validation_windows.features_df['secayo']


rf_classifier = RandomForestClassifier(n_estimators=100,bootstrap=True, random_state=42,max_depth=15,min_samples_split=5,min_samples_leaf=2,max_features='sqrt')
rf_classifier.fit(X_train, Y_train)
Y_pred = rf_classifier.predict(X_validation)

print(classification_report(Y_validation, Y_pred))
print(confusion_matrix(Y_validation, Y_pred))