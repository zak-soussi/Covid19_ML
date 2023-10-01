import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_row', 111)
pd.set_option('display.max_column', 111)

# Setting the Upload folder for the generated plots
UPLOAD_FOLDER = './Visualisations'

# Getting the dataset and determining the target column
dataset = pd.read_excel("./dataset.xlsx")
target = dataset["SARS-Cov-2 exam result"]
dataset = dataset.drop("SARS-Cov-2 exam result", axis=1)

# Feature types
dataset.dtypes.value_counts()

# Missing values
feature_presence = (dataset.isna().sum() / dataset.shape[0]).sort_values()
print(feature_presence)
sns.heatmap(dataset.isna(), cbar=False)
missing_values_path = f"{UPLOAD_FOLDER}/missing_values.png"
plt.savefig(missing_values_path, bbox_inches='tight')

# Target Visualization
target.value_counts().plot.pie()
target_path = f"{UPLOAD_FOLDER}/target.png"
plt.savefig(target_path, bbox_inches='tight')

# Removing unimportant features
delete_feature = [x for x in feature_presence.index if feature_presence[x] > 0.9]
delete_feature.append('Patient ID')
dataset = dataset.drop(delete_feature, axis=1)

# Visualisation des variables continues
float_features = [col for col in dataset.columns if dataset[col].dtype == 'float']
for i, feature in enumerate(float_features):
    sns.displot(x=feature, data=dataset, hue=target)
    continuos_feature_path = f"{UPLOAD_FOLDER}/continuos_feature_{i}.png"
    plt.savefig(continuos_feature_path, bbox_inches='tight')

# Visualisation des variables categorielles
df_object_features = dataset.select_dtypes('object')
for col in df_object_features:
    print(f'{col :-<40} {dataset[col].unique()}')
dataset.drop('Parainfluenza 2', axis=1, inplace=True)
df_object_features.drop('Parainfluenza 2', axis=1, inplace=True)
for i in range(4):
    plt.figure()
    sns.boxplot(x=df_object_features.columns[i], y=float_features[i], data=dataset, hue=target)
    categorial_continuous_path = f"{UPLOAD_FOLDER}/categorial_continuos_{i}.png"
    plt.savefig(categorial_continuous_path, bbox_inches='tight')
for i, col in enumerate(df_object_features):
    plt.figure()
    dataset[col].value_counts().plot.pie()
    categorial_feature_path = f"{UPLOAD_FOLDER}/categorial_feature_{i}.png"
    plt.savefig(categorial_feature_path, bbox_inches='tight')

# Divide our dataset
positive_dataset = dataset[target == 'positive']
negative_dataset = dataset[target == 'negative']
feature_presence = dataset.isna().sum() / dataset.shape[0]
feature_first_category = dataset.columns[(feature_presence < 0.9) & (feature_presence > 0.8)]
feature_second_category = dataset.columns[(feature_presence < 0.8) & (feature_presence > 0.7)]
