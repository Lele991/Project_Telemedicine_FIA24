import json
import numpy as np
import pandas as pd
import logging

from sklearn.preprocessing import MinMaxScaler

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_province_code_data(file_path):
    """
    Carica i dati delle province da un file JSON e restituisce due dizionari:
    - codice_to_provincia: Mappa i codici delle province ai nomi delle province.
    - provincia_to_codice: Mappa i nomi delle province ai codici delle province.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            province_list = data['Provincia']
    except FileNotFoundError:
        logging.error(f"Errore: Il file {file_path} non è stato trovato.")
        return None, None
    except json.JSONDecodeError:
        logging.error("Errore nella decodifica del file JSON.")
        return None, None

    # Dizionari per contenere sigla e provincia
    codice_to_provincia = {}
    provincia_to_codice = {}

    # Itera attraverso le province nel JSON e popola i dizionari
    for prov in province_list:
        codice = prov.get('desTarga')
        provincia = prov.get('desProvincia')
        
        if codice and provincia:
            codice_to_provincia[codice] = provincia
            provincia_to_codice[provincia] = codice

    logging.info(f"Province caricate con successo dal file {file_path}.")
    return codice_to_provincia, provincia_to_codice

def fetch_comuni_code_data(file_path):
    """
    Carica i dati dei comuni da un file JSON e restituisce due dizionari:
    - codice_to_comune: Mappa i codici dei comuni ai nomi dei comuni.
    - comune_to_codice: Mappa i nomi dei comuni ai codici dei comuni.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            comuni_list = data['Comune_StatoEstero_Cache']
    except FileNotFoundError:
        logging.error(f"Errore: Il file {file_path} non è stato trovato.")
        return None, None
    except json.JSONDecodeError:
        logging.error("Errore nella decodifica del file JSON.")
        return None, None

    # Dizionari per contenere sigla e comune
    codice_to_comune = {}
    comune_to_codice = {}

    # Itera attraverso i comuni nel JSON e popola i dizionari
    for com in comuni_list:
        codice = com.get('codIstat')
        comune = com.get('desComune')
        
        if codice and comune:
            # Rimuovi il primo carattere '0' da codIstat
            codice = codice[1:] if codice.startswith('0') else codice

            codice_to_comune[codice] = comune
            comune_to_codice[comune] = codice

    logging.info(f"Comuni caricati con successo dal file {file_path}.")
    return codice_to_comune, comune_to_codice

def process_province_comuni(dataset, codice_to_provincia, provincia_to_codice, codice_to_comune, comune_to_codice):
    """
    Processa il dataset per riempire i campi relativi a comuni e province
    utilizzando i dizionari di mappatura forniti.
    """
    # Controllo sui dati iniziali
    required_columns = ['comune_residenza', 'provincia_residenza', 'provincia_erogazione', 
                        'codice_comune_residenza', 'codice_provincia_residenza', 'codice_provincia_erogazione']
    
    missing_columns = [col for col in required_columns if col not in dataset.columns]
    if missing_columns:
        logging.error(f"Le seguenti colonne sono mancanti nel dataset: {missing_columns}")
        return dataset

    # Crea una copia esplicita del DataFrame
    dataset = dataset.copy()
    
    # Uniformare la capitalizzazione nel dataset e assegnare i risultati alle colonne
    dataset['comune_residenza'] = dataset['comune_residenza'].astype(str).str.upper()
    dataset['provincia_residenza'] = dataset['provincia_residenza'].astype(str).str.upper()
    dataset['provincia_erogazione'] = dataset['provincia_erogazione'].astype(str).str.upper()
    dataset['codice_comune_residenza'] = dataset['codice_comune_residenza'].astype(str).str.upper()
    dataset['codice_provincia_residenza'] = dataset['codice_provincia_residenza'].astype(str).str.upper()
    dataset['codice_provincia_erogazione'] = dataset['codice_provincia_erogazione'].astype(str).str.upper()

    # Uniformare la capitalizzazione nei dizionari di mappatura
    codice_to_provincia = {k.upper(): v for k, v in codice_to_provincia.items()}
    provincia_to_codice = {k.upper(): v for k, v in provincia_to_codice.items()}
    codice_to_comune = {k.upper(): v for k, v in codice_to_comune.items()}
    comune_to_codice = {k.upper(): v for k, v in comune_to_codice.items()}

    # Riempimento dei campi con le mappe
    dataset['comune_residenza'] = dataset['comune_residenza'].fillna(dataset['codice_comune_residenza'].map(codice_to_comune))
    dataset['provincia_residenza'] = dataset['provincia_residenza'].fillna(dataset['codice_provincia_residenza'].map(codice_to_provincia))
    dataset['provincia_erogazione'] = dataset['provincia_erogazione'].fillna(dataset['codice_provincia_erogazione'].map(codice_to_provincia))
    dataset['codice_comune_residenza'] = dataset['codice_comune_residenza'].fillna(dataset['comune_residenza'].map(comune_to_codice))
    dataset['codice_provincia_residenza'] = dataset['codice_provincia_residenza'].fillna(dataset['provincia_residenza'].map(provincia_to_codice))
    dataset['codice_provincia_erogazione'] = dataset['codice_provincia_erogazione'].fillna(dataset['provincia_erogazione'].map(provincia_to_codice))

    logging.info("Processamento delle province e comuni completato con successo.")
    return dataset

def fill_province_comuni(dataset, path_province, path_comuni):
    """
    Riempie i dati mancanti relativi a province e comuni nel dataset utilizzando i file JSON forniti.
    """
    # Carica i dati delle province e dei comuni
    codice_to_provincia, provincia_to_codice = fetch_province_code_data(path_province)
    codice_to_comune, comune_to_codice = fetch_comuni_code_data(path_comuni)

    # Controlla se i dizionari sono stati caricati correttamente
    if codice_to_provincia is not None and provincia_to_codice is not None and codice_to_comune is not None and comune_to_codice is not None:
        # Processa il dataset per riempire i campi mancanti
        dataset = process_province_comuni(dataset, codice_to_provincia, provincia_to_codice, codice_to_comune, comune_to_codice)
    else:
        logging.error("Errore nel caricamento dei dati di province o comuni. Il riempimento dei campi non è stato eseguito.")
    
    return dataset

def add_durata_visita(dataset):
    """
    Calcola la durata della visita per le righe.
    
    Aggiunge una nuova colonna 'durata_visita' che rappresenta la durata della visita in minuti.
    Gestisce i casi in cui 'ora_inizio_erogazione' o 'ora_fine_erogazione' sono mancanti o non validi.
    """
    # Controllo sui dati iniziali
    if 'ora_inizio_erogazione' not in dataset.columns or 'ora_fine_erogazione' not in dataset.columns:
        logging.error("Le colonne 'ora_inizio_erogazione' e 'ora_fine_erogazione' devono essere presenti nel dataset.")
        return dataset

    dataset['ora_inizio_erogazione'] = pd.to_datetime(dataset['ora_inizio_erogazione'], utc=True, errors='coerce')
    dataset['ora_fine_erogazione'] = pd.to_datetime(dataset['ora_fine_erogazione'], utc=True, errors='coerce')
    dataset['data_erogazione'] = pd.to_datetime(dataset['data_erogazione'], utc=True, errors='coerce')

    dataset['durata_visita'] = (dataset['ora_fine_erogazione'] - dataset['ora_inizio_erogazione']).dt.total_seconds() / 60

    # Validazione delle durate
    invalid_durations = dataset['durata_visita'] < 0
    if invalid_durations.any():
        logging.warning(f"Ci sono {invalid_durations.sum()} righe con durata visita negativa. Queste verranno impostate a NaN.")
    dataset['durata_visita'] = dataset['durata_visita'].apply(lambda x: np.nan if x < 0 else x)

    num_missing_durations = dataset['durata_visita'].isnull().sum()
    logging.info(f'Add Durata Visita: Calcolata la durata della visita. {num_missing_durations} righe hanno durate mancanti o non valide.')

    return dataset

def add_eta_paziente(dataset):
    """
    Calcola l'età del paziente in base alla data di nascita e aggiunge una nuova colonna 'eta_paziente'.
    Gestisce i casi in cui la data di nascita è mancante o non valida.
    """
    # Controllo sui dati iniziali
    if 'data_nascita' not in dataset.columns:
        logging.error("La colonna 'data_nascita' deve essere presente nel dataset.")
        return dataset

    dataset['data_nascita'] = pd.to_datetime(dataset['data_nascita'], errors='coerce')
    dataset['eta_paziente'] = (pd.to_datetime('today') - dataset['data_nascita']).dt.days // 365

    # Validazione dell'età
    invalid_ages = (dataset['eta_paziente'] < 0) | (dataset['eta_paziente'] > 120)
    if invalid_ages.any():
        logging.warning(f"Ci sono {invalid_ages.sum()} righe con età paziente non plausibile. Queste verranno impostate a NaN.")
    dataset['eta_paziente'] = dataset['eta_paziente'].apply(lambda x: np.nan if x < 0 or x > 120 else x)

    num_missing_ages = dataset['eta_paziente'].isnull().sum()
    logging.info(f'Add Eta Paziente: Calcolata l\'età del paziente. {num_missing_ages} righe hanno età mancanti o non valide.')

    # Rimozione della colonna 'data_nascita' dopo aver calcolato l'età
    dataset.drop(columns=['data_nascita'], inplace=True)

    return dataset

def categorizza_durata_visita(durata):
    if durata <= 15:
        return 'breve'
    elif 16 <= durata <= 30:
        return 'media'
    elif 31 <= durata <= 60:
        return 'lunga'
    elif 61 <= durata <= 87:
        return 'molto lunga'
    else:
        return 'fuori range'  # Per gestire eventuali valori imprevisti

def fill_durata_visita(dataset):
    """
    Calcola la durata della visita per le righe in cui 'durata_visita' è nulla.
    Per le righe mancanti di 'ora_inizio_erogazione' o 'ora_fine_erogazione',
    utilizza la media delle durate per il tipo di servizio offerto ('descrizione_attivita').
    """
    missing_durata = dataset['durata_visita'].isnull()

    # Calcola la durata della visita in minuti per le righe con 'durata_visita' mancante ma con orari disponibili
    dataset.loc[missing_durata, 'durata_visita'] = (
        dataset.loc[missing_durata, 'ora_fine_erogazione'] - 
        dataset.loc[missing_durata, 'ora_inizio_erogazione']
    ).dt.total_seconds() / 60

    # Gestisce i casi in cui mancano 'ora_inizio_erogazione' o 'ora_fine_erogazione'
    durata_media_per_servizio = dataset.groupby('descrizione_attivita')['durata_visita'].mean()

    dataset['durata_visita'] = dataset.apply(
        lambda row: durata_media_per_servizio[row['descrizione_attivita']] 
        if pd.isnull(row['durata_visita']) else row['durata_visita'], axis=1
    )

    num_filled_durations = dataset['durata_visita'].isnull().sum()
    logging.info(f'Fill Durata Visita: Riempite le durate mancanti. {num_filled_durations} righe hanno ancora durate mancanti.')

    if num_filled_durations > 0:
        logging.warning("Fill Durata Visita: Alcune durate della visita non sono state riempite. Controllare i dati.")
    else:
        dataset['durata_visita'] = dataset['durata_visita'].astype('int64')

    # dataset['classificazione_durata_visita'] = dataset['durata_visita'].apply(categorizza_durata_visita)

    return dataset

def add_fascia_eta_column(dataset):
    """
    Genera una Serie 'fascia_eta' basata sulla colonna 'eta_paziente' secondo le fasce specificate.
    """
    def determina_fascia_eta(eta):
        if eta < 12:
            return '0-12'
        elif eta < 24:
            return '13-24'
        elif eta < 36:
            return '25-36'
        elif eta < 48:
            return '37-48'
        elif eta < 60:
            return '49-60'
        elif eta < 70:
            return '61-70'
        else:
            return '71+'
    
    # Restituisce una nuova colonna (Serie) senza modificare direttamente il dataset
    dataset['fascia_eta'] = dataset['eta_paziente'].apply(determina_fascia_eta)

    # Normalizza i valori della colonna 'eta_paziente' tra 0 e 1
    scaler = MinMaxScaler()
    dataset['eta_paziente'] = scaler.fit_transform(dataset[['eta_paziente']])

    logging.info("Generata la colonna 'fascia_eta' basata sull'età del paziente.")
    return dataset


import pandas as pd

def colonne_to_category(df, colonne):
    """
    Converte le colonne specificate in 'category' per ottimizzare la memoria.
    
    Parametri:
    - df: Il DataFrame da ottimizzare.
    - colonne: Lista di nomi delle colonne da convertire in 'category'.
    
    Ritorna:
    - Il DataFrame con le colonne specificate convertite in 'category'.
    """
    for colonna in colonne:
        if colonna in df.columns:
            df[colonna] = df[colonna].astype('category')
            print(f"Colonna '{colonna}' convertita in 'category'.")
        else:
            print(f"Colonna '{colonna}' non trovata nel DataFrame.")
    
    # Controlla se 'codice_struttura_erogazione' può essere convertita in int
    if 'codice_struttura_erogazione' in df.columns and df['codice_struttura_erogazione'].apply(float.is_integer).all():
        df['codice_struttura_erogazione'] = df['codice_struttura_erogazione'].astype('int64')
        print("Colonna 'codice_struttura_erogazione' convertita in 'int64'.")

    return df
