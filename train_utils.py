from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.preprocessing import MinMaxScaler
from ucimlrepo import fetch_ucirepo
from imblearn.over_sampling import SMOTE
from pickle import dump
from pathlib import Path
import pandas as pd

DF_TARGETS = ["Diagnosis", "Management", "Severity"]

# Guarda o caminho completo para o diretório de modelos
MODELS_DIR = Path(__file__).resolve().parent / "models"

def get_dataframe():
    # Procura pelo dataset no repositório
    regensburg_pediatric_appendicitis = fetch_ucirepo(id=938)

    # Obtém os dataframes pandas
    X = regensburg_pediatric_appendicitis.data.features
    y = regensburg_pediatric_appendicitis.data.targets
    return pd.concat([X, y], axis=1) # Retorna o dataframe completo

def preprocess(df):
    columns_to_drop = []
    numerical_columns = []
    classes_columns = []

    # Separando as colunas em numéricas, categóricas e as que devem ser excluídas
    for column in df.columns:
        missing_perc = df[column].isnull().mean() * 100 # Percentagem de dados faltantes na coluna
        if  missing_perc > 40:
            columns_to_drop.append(column)
        elif pd.api.types.is_numeric_dtype(df[column]):
            numerical_columns.append(column)
        else:
            classes_columns.append(column)

    df = df.drop(columns=columns_to_drop, axis=1) # Remove as colunas designadas para exclusão

    # Preenche colunas numéricas com a mediana
    for column in numerical_columns:
        df[column] = df[column].fillna(df[column].median())

    # Preenche colunas categóricas com a moda
    for column in classes_columns:
        df[column] = df[column].fillna(df[column].mode()[0])
    return df

def normalize_data(df):
    df_targets = df[DF_TARGETS]

    df = df.drop(columns=df_targets) # Removendo colunas alvo

    # Isola as colunas numéricas
    numeric_df = df.select_dtypes(include='number')

    # Normaliza as colunas numéricas
    scaler = MinMaxScaler()
    df[numeric_df.columns] = scaler.fit_transform(numeric_df)

    # Junta novamente as colunas alvo de volta (reiniciando índice por razões de segurança)
    df.reset_index(drop=True, inplace=True)
    df_targets.reset_index(drop=True, inplace=True)
    df_normalized = pd.concat([df, df_targets], axis=1)

    with open(MODELS_DIR / "scaler_model.pkl", "wb") as f:
        dump(scaler, f)
    return df_normalized

def balance_column(df, target_column):
    df_features = df.drop(columns=DF_TARGETS, axis=1)
    df_classes = df[target_column]

    resampler = SMOTE(random_state=42)
    df_features_, df_classes_ = resampler.fit_resample(df_features, df_classes)
    return pd.concat([df_features_, df_classes_], axis=1) # Retorna o dataframe balanceado

def train_model(df, target_column):
    print(f"\nTraining model {target_column}...")

    # Separando atributos
    X = df.drop(columns=DF_TARGETS, axis=1)
    y = df[target_column]

    model = RandomForestClassifier(random_state=42)

    # Hiperparâmetros para Random Forest
    param_dist = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    # Otimização dos hiperparâmetros com validação cruzada
    randomized_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_distributions=param_dist,
        n_iter=10,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        random_state=42
    )

    # Modelo com melhores parâmetros
    best_model = randomized_search.best_estimator_

    print("Best parameters found:", randomized_search.best_params_)

    # Avalia o modelo final com validação cruzada
    scores = cross_validate(
        best_model, X, y,
        cv=5,
        scoring=["accuracy", "f1_macro"],
        return_train_score=False
    )

    # Imprime resultados importantes
    print("Cross validation scores computed:")
    print("Mean accuracy:", scores['test_accuracy'].mean())
    print("F1 Macro mean:", scores['test_f1_macro'].mean())

    # Salva o modelo treinado
    model_path = MODELS_DIR / f"{target_column.lower()}_model.pkl"
    with open(model_path, "wb") as f:
        dump(best_model, f)

def train_models():
    df = preprocess(get_dataframe())
    df = normalize_data(df) # Dados normalizados

    # Treina cada um dos modelos definidos
    for column in DF_TARGETS:
        balanced_df = balance_column(df, column)
        train_model(balanced_df, column)

