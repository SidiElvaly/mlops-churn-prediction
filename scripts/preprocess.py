#!/usr/bin/env python3
"""
scripts/preprocess.py

Étapes 2 à 5 de la branche data-pipeline :
 2. Analyse des types (numériques, catégorielles)
 3. Traitement des valeurs manquantes + encodage catégoriel
 4. Standardisation numérique
 5. Sauvegarde full & reduced
"""

import argparse
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def analyze_types(df: pd.DataFrame):
    print("\n>>> ANALYSE DES TYPES ET VALEURS MANQUANTES <<<")
    print("Types de colonnes :")
    print(df.dtypes, end="\n\n")
    print("Nombre de valeurs manquantes par colonne :")
    print(df.isna().sum(), end="\n\n")
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        print("Exemples de modalités (object) :")
        for c in obj_cols:
            vals = df[c].dropna().unique()[:5].tolist()
            print(f"  • {c} ({df[c].nunique()} modalités), ex. {vals}")
    print("-" * 50, "\n")

def clean_and_transform(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Convertir TotalCharges et drop NA
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    # 2. Imputation rapide par propagation (forward fill)
    df = df.ffill()

    # 3. Encodage des catégorielles
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ("customerID", "Churn")]
    encoder = OneHotEncoder(sparse_output=False, drop="first")
    cat_mat = encoder.fit_transform(df[cat_cols])
    cat_df  = pd.DataFrame(
        cat_mat,
        columns=encoder.get_feature_names_out(cat_cols),
        index=df.index
    )

    # 4. Standardisation des numériques
    num_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
    num_cols = [n for n in num_cols if n != "Churn"]
    scaler  = StandardScaler()
    num_mat = scaler.fit_transform(df[num_cols])
    num_df  = pd.DataFrame(num_mat, columns=num_cols, index=df.index)

    # 5. Concaténation et cible
    y = df["Churn"].map({"No": 0, "Yes": 1})
    out = pd.concat([ df[["customerID"]], num_df, cat_df ], axis=1)
    out["Churn"] = y
    return out

def save_dataframe(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)

def main():
    parser = argparse.ArgumentParser(
        description="Prétraitement Telco Churn (steps 2–5)"
    )
    parser.add_argument("--input",        required=True,
                        help="CSV raw (data/raw/...csv)")
    parser.add_argument("--output-full",  required=True,
                        help="Parquet full (data/processed/...)")
    parser.add_argument("--output-reduced", required=True,
                        help="Parquet reduced version")
    parser.add_argument("--reduced-frac", type=float, default=0.3,
                        help="Fraction pour reduced (entre 0 et 1)")
    args = parser.parse_args()

    # 1. Chargement
    df_raw = load_data(args.input)

    # 2. Analyse des types + missing  
    analyze_types(df_raw)

    # 3–4. Nettoyage, encodage, standardisation  
    df_clean = clean_and_transform(df_raw)

    # 5a. Sauvegarde full  
    save_dataframe(df_clean, args.output_full)
    print(f"✔️ Full data saved → {args.output_full}")

    # 5b. Sauvegarde reduced  
    df_red = df_clean.sample(frac=args.reduced_frac, random_state=42)
    save_dataframe(df_red, args.output_reduced)
    print(f"✔️ Reduced data ({int(100*args.reduced_frac)}%) saved → {args.output_reduced}")

if __name__ == "__main__":
    main()
