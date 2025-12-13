import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from IPython.display import display
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def classification_data_understanding(df, target):
    print("### Classification Data Understanding ###\n")

    print("📊 Distribution de la cible :")
    print(df[target].value_counts(normalize=True))

    plt.figure(figsize=(6,4))
    df[target].value_counts().plot(kind="bar")
    plt.title(f"Distribution de la cible : {target}")
    plt.show()

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    num_cols = num_cols.drop(target, errors="ignore")

    if len(num_cols) > 0:
        df[num_cols].hist(figsize=(14,10), bins=20)
        plt.suptitle("Histogrammes des features numériques")
        plt.show()

        plt.figure(figsize=(12,10))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm")
        plt.title("Matrice de corrélation")
        plt.show()

def regression_data_understanding(df, target):
    print("### Regression Data Understanding ###\n")

    print("📊 Distribution de la cible :")
    print(df[target].describe())

    plt.figure(figsize=(6,4))
    plt.hist(df[target], bins=20)
    plt.title(f"Distribution de la cible : {target}")
    plt.show()

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    num_features = num_cols.drop(target, errors="ignore")

    if len(num_features) > 0:
        df[num_features].hist(figsize=(14,10), bins=20)
        plt.suptitle("Histogrammes des features numériques")
        plt.show()

        # Corrélation features ↔ target
        corr = df[num_features].corrwith(df[target]).sort_values(ascending=False)

        print("📈 Corrélation avec la cible :")
        print(corr)

        plt.figure(figsize=(10,6))
        sns.barplot(x=corr.index, y=corr.values)
        plt.xticks(rotation=45)
        plt.title(f"Corrélation avec {target}")
        plt.show()

        # Matrice de corrélation entre features
        plt.figure(figsize=(12,10))
        sns.heatmap(df[num_features].corr(), annot=True, cmap="coolwarm")
        plt.title("Matrice de corrélation (features)")
        plt.show()

        plt.figure(figsize=(14,8))
        sns.boxplot(data=df[num_features], orient="h")
        plt.title("Boxplots des features numériques")
        plt.show()


def load_ucirepo_dataset(dataset_name, registry):
    if dataset_name not in registry:
        raise ValueError(f"Dataset '{dataset_name}' non défini dans le registre.")

    info = registry[dataset_name]

    dataset = fetch_ucirepo(id=info["id"])

    X = dataset.data.features.copy()
    y = dataset.data.targets

    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("Target multiple non supportée.")
        y = y.iloc[:, 0]

    X[info["target"]] = y

    return X, info["target"], info["task_type"]


# Registre des datasets UCI
UCI_DATASETS = {

    # -------------------------------
    # Regression
    # -------------------------------
    ##"boston_housing": {
        ##"id": 360,
        ##"target": "MEDV",
        ##"task_type": "regression"
    ##},

    "white_wine": {
        "id": 186,
        "target": "quality",
        "task_type": "regression"
    },

    "red_wine": {
        "id": 186,
        "target": "quality",
        "task_type": "regression"
    },

    "parkinson_oxford": {
        "id": 189,
        "target": "motor_UPDRS",
        "task_type": "regression"
    },

    "ozone": {
        "id": 172,
        "target": "Ozone",
        "task_type": "regression"
    },

    "concrete": {
        "id": 165,
        "target": "Concrete_compressive_strength",
        "task_type": "regression"
    },

    "auto_mpg": {
        "id": 9,
        "target": "mpg",
        "task_type": "regression"
    },

    # -------------------------------
    # Classification
    # -------------------------------
    ##"parkinson": {
      ##  "id": 174,
        ##"target": "status",
        ##"task_type": "classification"
    ##},

    "sonar": {
        "id": 151,
        "target": "Class",
        "task_type": "classification"
    },

    "breast_cancer": {
        "id": 17,
        "target": "diagnosis",
        "task_type": "classification"
    }
}

def handle_missing_outliers(df, method='median'):
    """
    Remplace les valeurs manquantes et traite les outliers.
    - df: DataFrame
    - method: 'median' ou 'mean' pour remplacer les NA
    """
    df_clean = df.copy()
    
    # Remplacer les valeurs manquantes
    for col in df_clean.select_dtypes(include=np.number).columns:
        if df_clean[col].isna().sum() > 0:
            fill_value = df_clean[col].median() if method=='median' else df_clean[col].mean()
            df_clean[col].fillna(fill_value, inplace=True)
    
    # Cap les outliers à 1.5*IQR
    for col in df_clean.select_dtypes(include=np.number).columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clean[col] = np.where(df_clean[col] < lower, lower, df_clean[col])
        df_clean[col] = np.where(df_clean[col] > upper, upper, df_clean[col])
    
    return df_clean

def prepare_regression_data(df, target, p_total=500, train_size=150, random_state=42):
    """
    Préparation des données pour la régression
    """
    df_clean = handle_missing_outliers(df)
    
    # Séparer X et y
    X = df_clean.drop(columns=[target])
    y = df_clean[target]
    
    # Standardisation
    numeric_cols = X.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    # Ajouter covariables artificielles jusqu'à p_total
    current_p = X.shape[1]
    while current_p < p_total:
        original_col = np.random.choice(numeric_cols)
        noise = np.random.normal(0, 1, size=X.shape[0])
        new_col = X[original_col] + noise
        X[f'artificial_{current_p}'] = new_col
        current_p += 1
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state, shuffle=True)
    
    return X_train, X_test, y_train, y_test

def prepare_classification_data(df, target, p_total=500, train_size=150, random_state=42):
    """
    Préparation des données pour la classification
    """
    df_clean = handle_missing_outliers(df)
    
    # Séparer X et y
    X = df_clean.drop(columns=[target])
    y = df_clean[target]
    
    # Standardisation
    numeric_cols = X.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    # Ajouter covariables artificielles jusqu'à p_total
    current_p = X.shape[1]
    while current_p < p_total:
        original_col = np.random.choice(numeric_cols)
        noise = np.random.normal(0, 1, size=X.shape[0])
        new_col = X[original_col] + noise
        X[f'artificial_{current_p}'] = new_col
        current_p += 1
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state, shuffle=True, stratify=y)
    
    return X_train, X_test, y_train, y_test

