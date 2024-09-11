
# Progetto: Clustering Supervisionato per la Teleassistenza

Questo progetto mira a profilare i pazienti in base al loro utilizzo del servizio di Teleassistenza, utilizzando tecniche di clustering supervisionato. L'obiettivo principale è identificare gruppi di pazienti con comportamenti simili, migliorando così la gestione e l'erogazione delle cure a distanza.

## Obiettivi

- Comprendere i fattori che determinano l'aumento dell'uso del servizio di Teleassistenza.
- Identificare pattern ricorrenti tra pazienti con malattie croniche.
- Fornire insight per migliorare la qualità del servizio e personalizzare le cure.
- Ridurre il carico sugli ospedali favorendo la deospedalizzazione.

## Struttura del Progetto

### Data
- **Sottocartella Italia**: Contiene file JSON con dati sui comuni, regioni e province italiane. Questi dati vengono utilizzati per creare le logiche di processamento.

### Data Preprocessing
La cartella **datapreprocessing** contiene file e sotto-cartelle per la preparazione dei dati utilizzati per l'analisi AI.

#### Sottocartelle
- **datafix**: Corregge errori o incongruenze nei dati.
- **datacleaner**: Pulisce i dati, eliminando duplicati e valori mancanti.

#### File Principali
- **clustering**: Organizza i dati in gruppi simili.
- **featureselection**: Seleziona le feature più rilevanti per il modello.
- **featureextractor**: Estrae nuove feature dai dati grezzi.
- **managedata**: Gestisce il dataset, integrando diverse fasi di preprocessamento.



### Graph
La cartella **graph** contiene grafici che visualizzano i risultati dei test effettuati, mostrando le performance dei modelli di clustering e selezione delle feature.

### Altri File
- **.gitignore**: Definisce i file e le cartelle da ignorare nel repository Git.
- **main**: Coordina tutte le operazioni, dal preprocessamento all'analisi dei dati.

### Gestione dei log del programma
**Logging**:
   - Lo script utilizza il logging per segnalare eventuali errori, come file non trovati o problemi nella formattazione dei dati, e per indicare il completamento del processo.


### Datacleaner.py

1. **remove_duplicates(dataset)**
   - Rimuove righe duplicate dal dataset.
   - Registra nel log quante righe duplicate sono state rimosse.

2. **remove_missing_values_rows(dataset, null_threshold=0.6)**
   - Rimuove le colonne che hanno una percentuale di valori nulli superiore a una soglia (default: 60%).
   - Registra nel log quali colonne sono state eliminate.

3. **remove_disdette(dataset)**
   - Rimuove le righe in cui `data_disdetta` non è nullo, ossia le cancellazioni.
   - Registra nel log quante righe sono state rimosse.

4. **remove_columns(dataset, columns)**
   - Elimina dal dataset le colonne specificate.

5. **handle_missing_values(dataset, strategy='mean')**
   - Riempie i valori mancanti in base alla strategia scelta (default: media).
   - Registra nel log il numero di valori mancanti trovati e gestiti.

6. **update_dataset_with_outliers(dataset, contamination=0.05, action='remove')**
   - Identifica e gestisce gli outlier con un approccio combinato tra **Isolation Forest** e **Local Outlier Factor**.
   - Gli outlier possono essere rimossi o marcati, a seconda dell'azione scelta.


### Datafix.py

1. **fetch_province_code_data(file_path)**:
   - Carica i dati delle province da un file JSON e restituisce dizionari che mappano i codici delle province ai nomi e viceversa.

2. **fetch_comuni_code_data(file_path)**:
   - Carica i dati dei comuni da un file JSON e restituisce dizionari che mappano i codici dei comuni ai nomi e viceversa.

3. **process_province_comuni(dataset, codice_to_provincia, provincia_to_codice, codice_to_comune, comune_to_codice)**:
   - Riempie i campi relativi a comuni e province nel dataset utilizzando i dizionari di mappatura caricati dai file JSON.

4. **add_durata_visita(dataset)**:
   - Calcola la durata della visita basandosi sull'ora di inizio e fine, aggiungendo una colonna 'durata_visita' in minuti.

5. **add_eta_paziente(dataset)**:
   - Calcola l'età del paziente in base alla data di nascita e aggiunge una colonna 'eta_paziente'. Le età non plausibili sono gestite come valori mancanti (NaN).

6. **fill_durata_visita(dataset)**:
   - Riempie i valori mancanti nella colonna 'durata_visita' utilizzando la durata media per tipo di servizio, o calcola la durata se disponibili ora di inizio e fine.

7. **add_fascia_eta_column(dataset)**:
   - Aggiunge una colonna 'fascia_eta' per categorizzare i pazienti in diverse fasce d'età e normalizza i valori di età.

8. **colonne_to_category(df, colonne)**:
   - Converte le colonne specificate in 'category' per ottimizzare la memoria.

#### Come Utilizzare:

1. **Dati di Input**:
   - Fornire i file JSON contenenti le informazioni su province e comuni, in modo che possano essere utilizzati per riempire i campi mancanti nel dataset.
   - Assicurarsi che il dataset contenga le colonne necessarie come `comune_residenza`, `provincia_residenza`, `ora_inizio_erogazione`, `data_nascita`, ecc.

2. **Processo**:
   - Utilizzare le funzioni descritte per caricare i dati dai file JSON e processare il dataset, riempiendo i valori mancanti e aggiungendo nuove colonne come 'durata_visita' ed 'eta_paziente'.


# Feature Selection

## Descrizione

La classe FeatureSelection è progettata per eseguire la selezione delle caratteristiche su un dataset, concentrandosi in particolare su colonne categoriali. L'algoritmo utilizza la V di Cramér per calcolare la correlazione tra le colonne e rimuove le caratteristiche che sono perfettamente o altamente correlate, in base a una soglia definita. Inoltre, genera heatmap per visualizzare la correlazione tra le variabili.

## Funzionalità Principali

- *Calcolo della correlazione tra variabili categoriali*: Utilizza la V di Cramér per determinare la correlazione tra le variabili categoriali.
- *Rimozione delle caratteristiche perfettamente correlate*: Rimuove le colonne con correlazione perfetta (V di Cramér pari a 1.0).
- *Rimozione delle caratteristiche altamente correlate*: Opzionalmente, rimuove le colonne con correlazione superiore a una soglia definita dall'utente.
- *Visualizzazione delle correlazioni*: Genera heatmap delle correlazioni prima e dopo il processo di selezione delle caratteristiche.
- *Pipeline completa*: Esegue l'intero processo di selezione delle caratteristiche in una singola chiamata.

## Comandi Principali

- **calculate_cramers_v(column1, column2)** :
Calcola il V di Cramér tra due variabili categoriali.
- **create_correlation_matrix()** :
Crea una matrice di correlazione utilizzando Cramér's V per tutte le colonne categoriali del dataset.
- **remove_perfectly_correlated_features(corr_matrix, threshold=1.0)**: Rimuove le colonne con correlazione perfetta.
- **remove_highly_correlated_features(corr_matrix, threshold=0.8)**:Rimuove le colonne con correlazione superiore alla soglia specificata.
- **display_heatmap(corr_matrix, title, filename)**: Genera e salva una heatmap della matrice di correlazione.
- **execute_feature_selection(threshold=0.8,remove_others_colum_by_threshold=False)**: Esegue l'intera pipeline di selezione delle caratteristiche.


## Output

•	I grafici saranno salvati nella directory ‘ graphs ‘ con il nome ‘ combined_plot.png ‘.

# Feature Extraction

## Descrizione

Questa classe **FeatureExtractor** è progettata per analizzare un dataset, identificare e categorizzare gli incrementi percentuali di servizi erogati in base al tempo (per trimestre e anno) e fornire una visualizzazione dei dati. La pipeline di analisi include il preprocessamento dei dati, il calcolo degli incrementi, la categorizzazione della crescita e la creazione di grafici per comprendere le tendenze del dataset.

## Funzionalità Principali

-	*Preprocessamento dei Dati*: Conversione delle date in formati utilizzabili per l'analisi e creazione di nuove colonne per anno e trimestre.
-	*Calcolo degli Incrementi Percentuali*: Raggruppamento dei dati per anno, trimestre e attività, calcolo della variazione percentuale del numero di servizi rispetto ai trimestri precedenti.
-	*Categorizzazione della Crescita*: Classificazione degli incrementi percentuali in categorie come 'crescita costante', 'crescita bassa', 'decrescita', ecc.
-	*Creazione di Grafici*: Generazione di istogrammi e boxplot per visualizzare la distribuzione degli incrementi percentuali, creazione di un grafico per analizzare l’andamento trimestrale delle teleassistenze.
-	*Pipeline Completa*: Esecuzione  dell'intera pipeline di analisi in sequenza.

## Comandi Principali

-	**preprocess_data()**:  Prepara i dati convertendo le date in formato datetime e creando le colonne "anno" e "trimestre".
-	**calculate_percentage_increments()**:  Calcola gli incrementi percentuali del numero di servizi per ogni attività, raggruppando per anno e trimestre.
-	**determine_growth_category(variazione_percentuale)**: Determina la categoria di crescita in base agli incrementi percentuali.
-	**apply_growth_categorization(grouped ’)**: Applica la categorizzazione della crescita al dataset originale.
-	**plot_graph()**:  Genera e salva grafici per visualizzare la distribuzione degli incrementi percentuali e l'andamento trimestrale delle teleassistenze.
-	**run_analysis()**:  Esegue l'intera pipeline di analisi dei dati, dal preprocessamento alla generazione dei grafici.



  ## Output

   - Heatmap delle correlazioni (iniziale e finale) salvate in graphs.
   -  Dataset ottimizzato senza variabili altamente correlate.
     
	 





