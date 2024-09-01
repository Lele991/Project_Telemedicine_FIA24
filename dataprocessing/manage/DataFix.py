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

def fill_province_comuni(dataset, codice_to_provincia, provincia_to_codice, codice_to_comune, comune_to_codice, log_file_path="mappature_log.txt"):
    # Apriamo il file di log per scrittura
    dataset['codice_comune_residenza'] = dataset['codice_comune_residenza'].replace('1168', 'None')
    with open(log_file_path, 'w') as log_file:
        # Scriviamo lo stato iniziale del dataset
        log_file.write("Stato iniziale del dataset:\n")
        log_file.write(str(dataset[['comune_residenza', 'codice_comune_residenza', 'provincia_residenza', 'codice_provincia_residenza']].head()) + "\n\n")

        # Riempimento del campo 'comune_residenza'
        log_file.write("Riempimento di 'comune_residenza' con la mappatura di 'codice_comune_residenza'...\n")
        dataset['comune_residenza'] = dataset['comune_residenza'].fillna(dataset['codice_comune_residenza'].map(codice_to_provincia))

        log_file.write("\nVerifica delle mappature di 'comune_residenza':\n")
        for idx, row in dataset[dataset['comune_residenza'].isna()].iterrows():
            log_file.write(f"Riga {idx}: codice_comune_residenza={row['codice_comune_residenza']} non ha trovato mappatura.\n")

        # Riempimento del campo 'provincia_residenza'
        log_file.write("\nRiempimento di 'provincia_residenza' con la mappatura di 'codice_provincia_residenza'...\n")
        dataset['provincia_residenza'] = dataset['provincia_residenza'].fillna(dataset['codice_provincia_residenza'].map(provincia_to_codice))

        log_file.write("\nVerifica delle mappature di 'provincia_residenza':\n")
        for idx, row in dataset[dataset['provincia_residenza'].isna()].iterrows():
            log_file.write(f"Riga {idx}: codice_provincia_residenza={row['codice_provincia_residenza']} non ha trovato mappatura.\n")

        # Riempimento del campo 'codice_comune_residenza'
        log_file.write("\nRiempimento di 'codice_comune_residenza' con la mappatura di 'comune_residenza'...\n")
        dataset['codice_comune_residenza'] = dataset['codice_comune_residenza'].fillna(dataset['comune_residenza'].map(comune_to_codice))

        log_file.write("\nVerifica delle mappature di 'codice_comune_residenza':\n")
        for idx, row in dataset[dataset['codice_comune_residenza'].isna()].iterrows():
            log_file.write(f"Riga {idx}: comune_residenza={row['comune_residenza']} non ha trovato mappatura.\n")

        # Riempimento del campo 'codice_provincia_residenza'
        log_file.write("\nRiempimento di 'codice_provincia_residenza' con la mappatura di 'provincia_residenza'...\n")
        dataset['codice_provincia_residenza'] = dataset['codice_provincia_residenza'].fillna(dataset['provincia_residenza'].map(codice_to_provincia))

        log_file.write("\nVerifica delle mappature di 'codice_provincia_residenza':\n")
        for idx, row in dataset[dataset['codice_provincia_residenza'].isna()].iterrows():
            log_file.write(f"Riga {idx}: provincia_residenza={row['provincia_residenza']} non ha trovato mappatura.\n")

        # Stato finale del dataset
        log_file.write("\nStato finale del dataset:\n")
        log_file.write(str(dataset[['comune_residenza', 'codice_comune_residenza', 'provincia_residenza', 'codice_provincia_residenza']].head()) + "\n")

    return dataset



