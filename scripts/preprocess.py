import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--output-data", type=str, default="/opt/ml/processing/output")
    args = parser.parse_args()

    input_path = os.path.join(args.input_data, "Titanic-Dataset.csv")
    
    if os.path.exists(input_path):
        df = pd.read_csv(input_path)
    else:
        print("Fichier non trouvé sur S3, téléchargement automatique du dataset Titanic...")
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        df = pd.read_csv(url)

    # --- FEATURE ENGINEERING TITANIC ---
    # 1. Traitement des valeurs manquantes
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna('S')

    # 2. Conversion des variables catégorielles en numériques (Encoding)
    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # 3. Sélection des caractéristiques (Features) utiles
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    target = 'Survived'

    df_clean = df[[target] + features]

    # --- SPLIT TRAIN / TEST ---
    train, test = train_test_split(df_clean, test_size=0.2, random_state=42)

    # Sauvegarde au format CSV sans en-tête ni index (requis pour SageMaker XGBoost)
    os.makedirs(os.path.join(args.output_data, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.output_data, "test"), exist_ok=True)

    train.to_csv(os.path.join(args.output_data, "train", "train.csv"), index=False, header=False)
    test.to_csv(os.path.join(args.output_data, "test", "test.csv"), index=False, header=False)

    print(f"Préparation Titanic terminée. Train: {train.shape}, Test: {test.shape}")