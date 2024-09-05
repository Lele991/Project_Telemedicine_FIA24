import pandas as pd
import numpy as np
import logging
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def remove_duplicates(dataset):
    """Rimuovi i duplicati nei dati."""
    before_shape = dataset.shape
    dataset.drop_duplicates(inplace=True)
    after_shape = dataset.shape
    logging.info(f'Remove Duplicates: {before_shape[0] - after_shape[0]} righe duplicate rimosse.')
    return dataset

def remove_missing_values_rows(dataset, null_threshold=0.6):
    """
    Rimuovi le colonne che hanno una percentuale di valori nulli superiore alla soglia specificata.
    
    Parametri:
    null_threshold (float): La soglia di nullità (da 0.0 a 1.0) oltre la quale una colonna viene rimossa.
                            Il valore predefinito è 0.6 (60%).
    
    Ritorna:
    dataset (pd.DataFrame): Il dataset pulito con le colonne rimosse.
    """
    before_shape = dataset.shape
    # Identifica le colonne da rimuovere
    columns_to_remove = dataset.columns[dataset.isnull().mean() >= null_threshold]
    # Rimuove le colonne identificate
    dataset = dataset.loc[:, dataset.isnull().mean() < null_threshold]
    after_shape = dataset.shape

    # Log migliorato con dettagli sulle colonne rimosse
    if len(columns_to_remove) > 0:
        logging.info(f"Remove Missing Values Rows: Rimosse {len(columns_to_remove)} colonne con troppi valori mancanti ({', '.join(columns_to_remove)}).")
    else:
        logging.info("Remove Missing Values Rows: Nessuna colonna rimossa.")

    return dataset

def remove_disdette(dataset):
    """
    Rimuove le righe con 'data_disdetta' non nullo.
    """
    if 'data_disdetta' not in dataset.columns:
        logging.error("La colonna 'data_disdetta' deve essere presente nel dataset.")
        return dataset

    before_shape = dataset.shape
    dataset = dataset[dataset['data_disdetta'].isnull()]
    after_shape = dataset.shape
    logging.info(f'Remove Disdette: {before_shape[0] - after_shape[0]} righe rimosse con data_disdetta non nullo.')
    return dataset

def remove_columns(dataset, columns):
    """
    Rimuovi dal dataset le colonne specificate.

    Parametri:
    columns (list): L'elenco delle colonne da rimuovere.
    """
    columns_to_remove = [col for col in columns if col in dataset.columns]
    dataset.drop(columns=columns_to_remove, inplace=True)
    logging.info(f'Remove Columns: Rimosse le colonne {columns_to_remove}.')
    return dataset

def handle_missing_values(dataset, strategy='mean'):
    """
    Gestisce i valori mancanti nel dataset, riempiendo i valori mancanti in base alla media (o altra misura) 

    Parametri:
    strategy (str): La strategia di riempimento ('mean', 'median', 'mode'). Default è 'mean'.
    """

    # Crea un oggetto SimpleImputer
    imputer = SimpleImputer(strategy=strategy)

    # Calcola la media per ciascun gruppo
    dataset_imputed = dataset.copy()
    missing_values = dataset_imputed.isnull().sum()
    total_missing = missing_values.sum()
    if total_missing > 0:
        logging.info(f"Handle Missing Values: Valori mancanti trovati in totale: {total_missing}")
        for col in dataset.columns:
            if dataset[col].isnull().any():
                missed_values = dataset[col].isnull().sum()
                logging(info=f'Handle Missing Values: Gestione dei valori mancanti per la colonna "{col}" con valori nulli "{missed_values}".')
                dataset_imputed[col] = imputer.fit_transform(dataset[[col]])
        logging.info(f'Handle Missing Values: Valori mancanti gestiti con strategia "{strategy}".')
    else:
        logging.info("Handle Missing Values: Nessun valore mancante trovato.")

    return dataset_imputed

def update_dataset_with_outliers(dataset, contamination=0.05, action='remove'):
    """
    Identifica e gestisce gli outlier usando un approccio ibrido che combina Isolation Forest e Local Outlier Factor.
    
    Parametri:
    contamination (float): La percentuale prevista di outlier nel dataset.
    action (str): Azione da intraprendere sugli outlier ('mark' per segnare, 'remove' per rimuovere).
    """
    n_estimators = 100
    max_samples = 'auto'
    n_jobs = -1  # Usa tutti i core disponibili
    
    # Considera un subset di colonne rilevanti
    relevant_columns = ['eta_paziente', 'durata_visita', 'codice_descrizione_attivita']

    # Codifica le variabili categoriali in numeriche
    dataset_encoded = pd.get_dummies(dataset[relevant_columns], drop_first=True)
    
    # Step 1: Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, n_estimators=n_estimators, max_samples=max_samples, n_jobs=n_jobs, random_state=42)
    iso_outliers = iso_forest.fit_predict(dataset_encoded)

    # Step 2: Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination, n_jobs=n_jobs)
    lof_outliers = lof.fit_predict(dataset_encoded)
    
    # Combina i risultati (condizione AND)
    outliers = ((iso_outliers == -1) & (lof_outliers == -1)).astype(int)

    # Aggiungi i risultati al dataset
    dataset['outlier'] = outliers

    # Considera età <= 0 o >= 100 come outlier
    dataset.loc[(dataset['eta_paziente'] <= 0) | (dataset['eta_paziente'] >= 100), 'outlier'] = -1

    num_outliers = (dataset['outlier'] == -1).sum()
    
    if action == 'remove':
        dataset = dataset[dataset['outlier'] == 0].copy()
        logging.info(f'Outliers: Rimossi {num_outliers} outlier.')
    else:  # Default action is 'mark'
        logging.info(f'Outliers: Segnati {num_outliers} righe come outlier.')
    
    return dataset
