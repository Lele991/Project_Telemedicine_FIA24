# metodo per riempire le province/comuni mancanti

import json

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

def fill_province_comuni(dataset, codice_to_provincia, provincia_to_codice, codice_to_comune, comune_to_codice):
    # Uniformare la capitalizzazione nel dataset
    dataset['comune_residenza'] = dataset['comune_residenza'].str.upper()
    dataset['provincia_residenza'] = dataset['provincia_residenza'].str.upper()
    dataset['codice_comune_residenza'] = dataset['codice_comune_residenza'].astype(str).str.upper()
    dataset['codice_provincia_residenza'] = dataset['codice_provincia_residenza'].str.upper()

    # Uniformare la capitalizzazione nei dizionari di mappatura
    codice_to_provincia = {k.upper(): v for k, v in codice_to_provincia.items()}
    provincia_to_codice = {k.upper(): v for k, v in provincia_to_codice.items()}
    codice_to_comune = {k.upper(): v for k, v in codice_to_comune.items()}
    comune_to_codice = {k.upper(): v for k, v in comune_to_codice.items()}

    # Riempimento del campo 'comune_residenza'
    dataset['comune_residenza'] = dataset['comune_residenza'].fillna(dataset['codice_comune_residenza'].map(codice_to_comune))

    # Riempimento del campo 'provincia_residenza'
    dataset['provincia_residenza'] = dataset['provincia_residenza'].fillna(dataset['codice_provincia_residenza'].map(codice_to_provincia))

    # Riempimento del campo 'codice_comune_residenza'
    dataset['codice_comune_residenza'] = dataset['codice_comune_residenza'].fillna(dataset['comune_residenza'].map(comune_to_codice))

    # Riempimento del campo 'codice_provincia_residenza'
    dataset['codice_provincia_residenza'] = dataset['codice_provincia_residenza'].fillna(dataset['provincia_residenza'].map(provincia_to_codice))

    return dataset
