# metodo per riempire le province/comuni mancanti

import json
import pandas as pd

def fetch_province_code_data(file_path):
    # Carica il file JSON locale
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            province_list = data['Provincia']
    except FileNotFoundError:
        print(f"Errore: Il file {file_path} non è stato trovato.")
        return None, None
    except json.JSONDecodeError:
        print("Errore nella decodifica del file JSON.")
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

    return codice_to_provincia, provincia_to_codice

def fetch_comuni_code_data(file_path):
    # Carica il file JSON locale
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            comuni_list = data['Comune_StatoEstero_Cache']
    except FileNotFoundError:
        print(f"Errore: Il file {file_path} non è stato trovato.")
        return None, None
    except json.JSONDecodeError:
        print("Errore nella decodifica del file JSON.")
        return None, None

    # Dizionari per contenere sigla e provincia
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

    return codice_to_comune, comune_to_codice

import pandas as pd

import pandas as pd

def process_province_comuni(dataset, codice_to_provincia, provincia_to_codice, codice_to_comune, comune_to_codice):
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

    return dataset



def fill_province_comuni(dataset, path_province, path_comuni):
    # Call the fetch_province_code_data and fetch_comuni_code_data function from DataFix.py
    # to get the dictionaries codice_to_provincia and provincia_to_codice
    codice_to_provincia, provincia_to_codice = fetch_province_code_data(path_province)
    codice_to_comune, comune_to_codice = fetch_comuni_code_data(path_comuni)

    # Check if the dictionaries are not None
    if codice_to_provincia is not None and provincia_to_codice is not None and codice_to_comune is not None and comune_to_codice is not None:
        # Call the process_province_comuni
        dataset = process_province_comuni(dataset, codice_to_provincia, provincia_to_codice, codice_to_comune, comune_to_codice)
    return dataset

def add_durata_visita(dataset):
    """
    Calcola la durata della visita per le righe.
    
    Aggiunge una nuova colonna 'durata_visita' che rappresenta la durata della visita in minuti.
    """
    
    # Converte le colonne in datetime
    dataset['ora_inizio_erogazione'] = pd.to_datetime(dataset['ora_inizio_erogazione'], utc=True, errors='coerce')
    dataset['ora_fine_erogazione'] = pd.to_datetime(dataset['ora_fine_erogazione'], utc=True, errors='coerce')

    # Calcola la durata della visita in minuti per le righe rimanenti
    dataset['durata_visita'] = (dataset['ora_fine_erogazione'] - dataset['ora_inizio_erogazione']).dt.total_seconds() / 60
    
    return dataset

def add_eta_paziente(dataset):
    """
    Calcola l'età del paziente in base alla data di nascita e aggiunge una nuova colonna 'eta_paziente'.
    """
    # Converte la colonna 'data_nascita' in datetime
    dataset['data_nascita'] = pd.to_datetime(dataset['data_nascita'], errors='coerce')

    # Calcola l'età del paziente in base alla data di nascita
    dataset['eta_paziente'] = (pd.to_datetime('today') - dataset['data_nascita']).dt.days // 365

    # Rimuove la colonna 'data_nascita'
    dataset.drop(columns=['data_nascita'], inplace=True)

    return dataset

def fill_durata_visita(dataset):
    """
    Calcola la durata della visita per le righe in cui 'durata_visita' è nulla.
    """
    # Converte le colonne in datetime
    dataset['ora_inizio_erogazione'] = pd.to_datetime(dataset['ora_inizio_erogazione'], utc=True)
    dataset['ora_fine_erogazione'] = pd.to_datetime(dataset['ora_fine_erogazione'], utc=True)
    
    # Trova le righe dove 'durata_visita' è nulla
    missing_durata = dataset['durata_visita'].isnull()

    # Calcola la durata della visita in minuti per le righe con 'durata_visita' mancante
    dataset.loc[missing_durata, 'durata_visita'] = (
        dataset.loc[missing_durata, 'ora_fine_erogazione'] - 
        dataset.loc[missing_durata, 'ora_inizio_erogazione']
    ).dt.total_seconds() / 60
    
    return dataset
