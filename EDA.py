import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

pd.set_option('display.max_row', 111)
pd.set_option('display.max_column', 111)

# Setting the Upload folder for some examples of the generated plots
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

# Removing unimportant features
delete_feature = [x for x in feature_presence.index if feature_presence[x] > 0.9]
delete_feature.append('Patient ID')
dataset = dataset.drop(delete_feature, axis=1)

# Visualisation des variables continues
visualisation_exp = 0
float_features = [col for col in dataset.columns if dataset[col].dtype == 'float']
for i, feature in enumerate(float_features):
    sns.displot(x=feature, data=dataset, hue=target)
    if visualisation_exp < 2:
        continuos_feature_path = f"{UPLOAD_FOLDER}/continuos_feature_{i}.png"
        plt.savefig(continuos_feature_path, bbox_inches='tight')
        visualisation_exp += 1

# Visualisation des variables categorielles
df_object_features = dataset.select_dtypes('object')
for col in df_object_features:
    print(f'{col :-<40} {dataset[col].unique()}')
dataset.drop('Parainfluenza 2', axis=1, inplace=True)
df_object_features.drop('Parainfluenza 2', axis=1, inplace=True)

visualisation_exp = 0
for i in range(4):
    plt.figure()
    sns.boxplot(x=df_object_features.columns[i], y=float_features[i], data=dataset, hue=target)
    if visualisation_exp == 0 :
        categorial_continuous_path = f"{UPLOAD_FOLDER}/categorial_continuos_{i}.png"
        plt.savefig(categorial_continuous_path, bbox_inches='tight')
        visualisation_exp +=1

visualisation_exp = 0
for i, col in enumerate(df_object_features):
    plt.figure()
    dataset[col].value_counts().plot.pie()
    if visualisation_exp < 2:
        categorial_feature_path = f"{UPLOAD_FOLDER}/categorial_feature_{i}.png"
        plt.savefig(categorial_feature_path, bbox_inches='tight')
        visualisation_exp += 1

# Divide our features into groups
feature_presence = dataset.isna().sum() / dataset.shape[0]
feature_continuos_category = dataset.columns[(feature_presence < 0.9) & (feature_presence > 0.88)]
feature_categorial_category = dataset.columns[(feature_presence < 0.8) & (feature_presence > 0.7)]


# Relation target / features
visualisation_exp = 0
for i, col in enumerate(feature_continuos_category):
    plt.figure()
    sns.displot(x=col, data=dataset, hue=target, kind='kde')
    if visualisation_exp < 2:
        target_continuous_path = f"{UPLOAD_FOLDER}/target_continuos_{i}.png"
        plt.savefig(target_continuous_path, bbox_inches='tight')
        visualisation_exp += 1

visualisation_exp = 0
for i, col in enumerate(feature_categorial_category):
    plt.figure()
    sns.heatmap(pd.crosstab(target, dataset[col]), annot=True, fmt='d')
    if visualisation_exp < 2:
        target_categorial_path = f"{UPLOAD_FOLDER}/target_categorial_{i}.png"
        plt.savefig(target_categorial_path, bbox_inches='tight')
        visualisation_exp += 1

# Relation between continuos_features
sns.clustermap(dataset[feature_continuos_category].corr())
continuos_continuos_path = f"{UPLOAD_FOLDER}/continuos_continuos.png"
plt.savefig(continuos_continuos_path, bbox_inches='tight')

# Relation between numerical_features
correlation_dataset = dataset.corr()
for col in dataset.select_dtypes('int'):
    print(f'Feature: {col}')
    print(correlation_dataset[col].sort_values(ascending=False))
    print('-' * 50)

# How many maladies does a patient have?
dataset['maladies_count'] = (dataset[feature_categorial_category] == 'detected').sum(axis=1)
sns.countplot(x='maladies_count', data=dataset)

# Exploring Nan values in depth
continuos_dataset = dataset[feature_continuos_category]
continuos_dataset['covid'] = target
print('Target distribution in continuos_dataset')
print(continuos_dataset.dropna()['covid'].value_counts(normalize=True))
print('-' * 50)
categorial_dataset = dataset[feature_categorial_category]
categorial_dataset['covid'] = target
print('Target distribution in categorial_dataset')
print(categorial_dataset.dropna()['covid'].value_counts(normalize=True))

# T-Test

positive_dataset = dataset[target == 'positive']
negative_dataset = dataset[target == 'negative']
negative_dataset = negative_dataset.sample(positive_dataset.shape[0])


def t_test(col):
    limit = 0.02
    stat, p = ttest_ind(positive_dataset[col].dropna(), negative_dataset[col].dropna())
    if p < limit:
        return "HO à rejeter"
    else:
        return "0"


for col in feature_continuos_category:
    print(f'{col :-<50}{t_test(col)}')
