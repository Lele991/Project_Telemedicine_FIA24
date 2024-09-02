import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.impute import SimpleImputer

def remove_duplicates(dataset):
    """Rimuovi i duplicati nei dati."""
    dataset.drop_duplicates(inplace=True)
    return dataset

def remove_missing_values_rows(dataset, null_threshold=0.6):
    """
    Rimuovi le colonne che hanno una percentuale di valori nulli superiore alla soglia specificata
    e rimuovi colonne specifiche se tutti i valori sono nulli.
    
    Parametri:
    null_threshold (float): La soglia di nullità (da 0.0 a 1.0) oltre la quale una colonna viene rimossa.
                            Il valore predefinito è 0.6 (60%).
    """
    # Rimuovi colonne con troppi valori nulli
    dataset = dataset.loc[:, dataset.isnull().mean() < null_threshold]

    return dataset

def remove_disdette(dataset): 
    """
    Rimuove le righe con 'data_disdetta' non nullo.
    Parametri:
    dataset (pd.DataFrame): Il DataFrame da cui rimuovere le righe con 'data_disdetta' non nullo.
    """

    dataset = dataset[dataset['data_disdetta'].isnull()]

    return dataset

def remove_columns(dataset, columns):
    """
    Rimuovi dal dataset le colonne specificate.

    Parametri:
    - dataset (pd.DataFrame): Il dataframe da cui rimuovere le colonne.
    - unimportant_columns (list): L'elenco delle colonne da rimuovere.
    """

    # Controlla se le colonne esistono nel dataset prima di rimuoverle
    columns_to_remove = [col for col in columns if col in dataset.columns]
    
    # Rimuovi le colonne
    dataset.drop(columns=columns_to_remove, inplace=True)

    return dataset

def update_dataset_with_outliers(dataset, eps=0.5, min_samples=5):
    """
    Aggiorna il dataset identificando gli outlier usando l'algoritmo DBSCAN.
    
    Parametri:
    eps (float): Il parametro di distanza massima tra due punti per essere considerati nello stesso cluster.
    min_samples (int): Il numero minimo di punti richiesti per formare un cluster.
    """
    # Informazioni prima del clustering
    print("Informazioni prima del clustering:")
    print(f"Numero di righe nel dataset: {dataset.shape[0]}")
    print(f"Numero di valori NaN in 'eta_paziente': {dataset['eta_paziente'].isna().sum()}")
    print(f"Numero di valori NaN in 'durata_visita': {dataset['durata_visita'].isna().sum()}")

    # Seleziona solo le colonne numeriche per il clustering
    numeric_data = dataset[['eta_paziente', 'durata_visita']].copy()

    # Imputa i valori NaN con la media
    imputer = SimpleImputer(strategy='mean')
    numeric_data_imputed = imputer.fit_transform(numeric_data)
    
    # Inizializza e applica DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(numeric_data_imputed)
    
    # Aggiungi colonne al dataset originale
    dataset['cluster'] = clusters
    dataset['outlier'] = (clusters == -1).astype(int)
    
    # Considera le righe con età > 100 come outlier (opzionale)
    dataset.loc[dataset['eta_paziente'] > 100, 'outlier'] = 1

    # Copia del dataset pulito
    cleaned_dataset = dataset[dataset['outlier'] == 0].copy()

    # Informazioni dopo il clustering
    print("\nInformazioni dopo il clustering:")
    print(f"Numero di outlier identificati: {dataset['outlier'].sum()}")
    print(f"Numero di righe nel dataset pulito: {cleaned_dataset.shape[0]}")
    print(f"Numero di valori NaN in 'eta_paziente' nel dataset pulito: {cleaned_dataset['eta_paziente'].isna().sum()}")
    print(f"Numero di valori NaN in 'durata_visita' nel dataset pulito: {cleaned_dataset['durata_visita'].isna().sum()}")

    return cleaned_dataset


