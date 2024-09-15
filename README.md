
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

#### Sottocartella manage
- **datafix**: Corregge errori o incongruenze nei dati.
- **datacleaner**: Pulisce i dati, eliminando duplicati e valori mancanti.
- **dataplot** : Si occupa di generare e salvare grafici che visualizzano la distribuzione dei cluster.

#### File Principali
- **clustering**: Organizza i dati in gruppi simili.
- **featureselection**: Seleziona le feature più rilevanti per il modello.
- **featureextractor**: Estrae nuove feature dai dati grezzi.
- **managedata**: Gestisce il dataset, integrando diverse fasi di preprocessamento.



### Graphs
La cartella **graphs** contiene grafici che visualizzano i risultati dei test effettuati, mostrando le performance dei modelli di clustering e selezione delle feature.

### Saved_models
La cartella **Saved_models** contiene un file pickle con un modello di clustering KMeans salvato.

### Altri File
- **.gitignore**: Definisce i file e le cartelle da ignorare nel repository Git.
- **main**: Coordina tutte le operazioni, dal preprocessamento all'analisi dei dati.

### Gestione dei log del programma
**Logging**:
   - Lo script utilizza il logging per segnalare eventuali errori, come file non trovati o problemi nella formattazione dei dati, e per indicare il completamento del processo.

### ManageData.py

La classe `ManageData` gestisce l'intero processo di preprocessing, analisi e clustering del dataset per l'analisi dei servizi di teleassistenza. 

#### Funzionalità Principali:

1. **replace_none_with_nan**: Sostituisce i valori 'None' e `None` con `NaN` all'interno del dataset per standardizzare i valori mancanti.

2. **log_missing_values**: Logga le colonne del dataset che contengono valori mancanti, fornendo un conteggio dettagliato per ogni colonna.

3. **save_dataset**: Salva il dataset processato in formato Parquet.

4. **print_columns**: Logga e stampa l'elenco delle colonne presenti nel dataset.

5. **clean_data**: Esegue un ciclo completo di pulizia del dataset:
    - Rimuove cancellazioni (`data_disdetta` non nullo).
    - Riempie i campi comuni e province utilizzando dati esterni.
    - Rimuove duplicati e colonne non necessarie.
    - Gestisce i valori mancanti in base a una soglia di tolleranza configurabile.
      
6. **run_analysis**: Esegue il ciclo completo di analisi, che include:
    - Pulizia del dataset.
    - Aggiunta della durata della visita e calcolo dell'età del paziente.
    - Identificazione e gestione degli outlier.
    - Selezione delle feature più rilevanti attraverso la **FeatureSelection**.
    - Estrazione di nuove feature attraverso la **FeatureExtractor**.
    - Applicazione del clustering attraverso la classe **Clustering** e salvataggio del dataset con l'aggiunta delle etichette di cluster.
    - Visualizzazione e salvataggio dei grafici tramite la classe **DataPlot**.

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


### DataPlot,py


#### Metodi Principali

1. **save_plot(self, plt, filename)**: Salva il grafico generato nella directory 'graphs'.

2. **plot_cluster_distribution(self)**: 
   - Crea un grafico a barre per visualizzare la distribuzione dei cluster nel dataset.

3. **plot_sex_distribution_by_cluster(self)**: 
   - Visualizza la distribuzione di genere nei vari cluster.

4. **plot_age_distribution_by_cluster(self)**: 
   - Mostra la distribuzione delle fasce di età nei cluster.

5. **plot_visit_duration_by_cluster(self)**: 
   - Visualizza la durata media delle visite per ciascun cluster.

6. **plot_cluster_by_region(self)**: 
   - Mostra la distribuzione dei cluster per regione di residenza.

7. **plot_cluster_by_structure_type(self)**: 
   - Visualizza la distribuzione dei cluster per tipologia di struttura di erogazione.

8. **plot_increment_by_cluster(self)**: 
   - Mostra la distribuzione degli incrementi classificati per ciascun cluster.

9. **plot_cluster_by_quarter(self)**: 
   - Visualizza la distribuzione dei cluster per trimestre.

10. **generate_plots(self)**: 
    - Esegue tutti i grafici disponibili e li salva nella directory 'graphs'.

## Output

Tutti i grafici sono salvati nella directory 'graphs'. I principali grafici includono:
- **Distribuzione dei Cluster** (`cluster_distribution.png`)
  
- **Distribuzione del Sesso nei Cluster** (`sex_distribution_by_cluster.png`)
  
- **Distribuzione delle Fasce d'Età nei Cluster** (`age_distribution_by_cluster.png`)
  
- **Durata delle Visite per Cluster** (`visit_duration_by_cluster.png`)
  
- **Distribuzione dei Cluster per Regione** (`cluster_by_region.png`)
  
- **Distribuzione dei Cluster per Tipologia di Struttura** (`cluster_by_structure_type.png`)
  
- **Incremento Classificato per Cluster** (`increment_by_cluster.png`)
  
- **Distribuzione dei Cluster per Trimestre** (`cluster_by_quarter.png`)

### Clustering.py

- **n_clusters**: Numero di cluster per KModes.
- **use_one_hot**: Se impostato su True, utilizza One-Hot Encoding per le variabili categoriali, altrimenti utilizza Label Encoding.

#### Metodi Principali

1. **get_dataset_clustered()**: Restituisce il dataset con le etichette dei cluster.
   
2. **elbow_method(self, dataset, min_clusters=4, max_clusters=10)**: Esegue l'Elbow Method per determinare il numero ottimale di cluster e salva il grafico.

3. **preprocess_data(self, dataset)**: Trasforma le variabili categoriali usando One-Hot o Label Encoding, e standardizza i dati per il clustering.

4. **fit(self, dataset)**: Esegue il clustering con KModes, calcola il silhouette score, e restituisce il dataset con le etichette dei cluster.

5. **calculate_purity(self, dataset, label_column='incremento_classificato')**: Calcola la purezza del clustering basata su una colonna di riferimento.

6. **plot_clusters(self, dataset)**: Crea e salva un grafico 2D e un silhouette plot dei cluster.

7. **plot_clusters_3d(self, dataset)**: Crea e salva un grafico 3D dei cluster utilizzando PCA.

8. **run_clustering(self, dataset, label_column='incremento_classificato', excluded_columns=None)**: Esegue l'intero processo di clustering, inclusi clustering, calcolo della purezza, plotting e salvataggio dei risultati.

## Output

- **Grafici Salvati**: Tutti i grafici sono salvati nella directory `graphs`:
  - **Elbow Method** (`elbow_method.png`)
  - **PCA Plot 2D** (`pca_clusters.png`)
  - **PCA Plot 3D** (`pca_clusters_3d.png`)
  - **Silhouette Plot** (`silhouette_plot.png`)

- **Risultati Clustering**: I risultati del clustering vengono salvati nella directory `results` in formato JSON:
  - **clustering_results.json**: Contiene le informazioni su colonne escluse/utilizzate, numero di cluster, silhouette score medio, purezza e metrica finale.


### FeatureSelection.py

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

### FeatureExtraction.py

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
     
	 





