
### ManageData.py

1. **`__init__(self, dataset, path_province, path_comuni, missing_threshold=0.6)`**:
   - Inizializza l'oggetto ManageData con il dataset, i percorsi ai file delle province e dei comuni, e una soglia per i valori mancanti.
   - Registra nel log l'inizio del processo di gestione dei dati.

2. **`get_dataset(self)`**:
   - Restituisce il dataset attualmente in uso.

3. **`set_dataset(self, dataset)`**:
   - Imposta un nuovo dataset.

4. **`replace_none_with_nan(self, dataset)`**:
   - Sostituisce i valori 'None' e None con NaN nel DataFrame.
   - Registra nel log la sostituzione eseguita.

5. **`log_missing_values(self, dataset)`**:
   - Registra nel log le colonne contenenti valori mancanti nel dataset.

6. **`save_dataset(self, dataset, name='extractor_dataset')`**:
   - Salva il dataset in formato Parquet.
   - Registra nel log il percorso del file salvato.

7. **`print_columns(self, dataset)`**:
   - Stampa le colonne del dataset nel log.

8. **`clean_data(self)`**:
   - Esegue una pulizia completa del dataset:
     - Sostituisce i valori None con NaN.
     - Rimuove righe con 'data_disdetta' non nullo.
     - Riempie province e comuni mancanti utilizzando DataFix.
     - Rimuove duplicati.
     - Rimuove colonne con valori mancanti sopra la soglia specificata.
     - Registra ogni operazione di pulizia nel log.

9. **`run_analysis(self)`**:
   - Esegue un'analisi completa dei dati:
     - Pulizia del dataset (esegue il metodo `clean_data`).
     - Aggiunge la durata della visita e riempie i valori mancanti.
     - Aggiunge l'età e la fascia d'età del paziente.
     - Identifica e gestisce gli outlier.
     - Rimuove colonne non necessarie.
     - Gestisce i valori mancanti.
     - Esegue la selezione delle caratteristiche utilizzando il modulo FeatureSelection.
     - Esegue l'estrazione delle caratteristiche utilizzando FeatureExtractor.
     - Salva il dataset elaborato e visualizza le colonne finali.
     - Esegue il clustering dei dati utilizzando KModes.
     - Genera e salva i grafici finali utilizzando DataPlot.
    


### DataFix.py

1. **fetch_province_code_data(file_path)**:
   - Carica i dati delle province da un file JSON.
   - Restituisce due dizionari: `codice_to_provincia` e `provincia_to_codice`, che mappano i codici delle province ai nomi e viceversa.
   - Registra nel log se il file non è trovato o se ci sono errori di decodifica.

2. **fetch_comuni_code_data(file_path)**:
   - Carica i dati dei comuni da un file JSON.
   - Restituisce due dizionari: `codice_to_comune` e `comune_to_codice`, che mappano i codici dei comuni ai nomi e viceversa.
   - Registra nel log se il file non è trovato o se ci sono errori di decodifica.

3. **process_province_comuni(dataset, codice_to_provincia, provincia_to_codice, codice_to_comune, comune_to_codice)**:
   - Processa il dataset per riempire i campi relativi ai comuni e alle province utilizzando i dizionari di mappatura.
   - Registra nel log se ci sono colonne mancanti nel dataset.

4. **fill_province_comuni(dataset, path_province, path_comuni)**:
   - Riempie i dati mancanti relativi alle province e ai comuni nel dataset utilizzando i file JSON forniti.
   - Registra nel log eventuali errori di caricamento dei dati.

5. **add_durata_visita(dataset)**:
   - Calcola la durata della visita per ciascuna riga, aggiungendo una colonna `durata_visita` che rappresenta la durata in minuti.
   - Gestisce casi in cui le ore di inizio o fine erogazione non sono valide o mancanti, e registra tali righe nel log.

6. **add_eta_paziente(dataset)**:
   - Calcola l'età del paziente basata sulla data di nascita e aggiunge una colonna `eta_paziente`.
   - Rimuove valori non plausibili di età e registra il numero di valori invalidi nel log.

7. **fill_durata_visita(dataset)**:
   - Riempie le durate di visita mancanti utilizzando la media delle durate per ciascun tipo di servizio.
   - Registra nel log quante righe hanno ancora durate mancanti.

8. **add_fascia_eta_column(dataset)**:
   - Genera una nuova colonna `fascia_eta` basata sull'età del paziente.
   - Normalizza la colonna `eta_paziente` tra 0 e 1 utilizzando MinMaxScaler.

9. **colonne_to_category(df, colonne)**:
   - Converte le colonne specificate in 'category' per ottimizzare la memoria.
   - Se possibile, converte anche la colonna `codice_struttura_erogazione` in `int64`.


