# Progetto: Clustering Supervisionato per la Teleassistenza
Questo progetto mira a profilare i pazienti in base al loro utilizzo del servizio di Teleassistenza, utilizzando tecniche di clustering supervisionato. L'obiettivo principale è identificare gruppi di pazienti con comportamenti simili, migliorando così la gestione e l'erogazione delle cure a distanza.

## Obiettivi
- Comprendere i fattori che determinano l'aumento dell'uso del servizio di Teleassistenza.
- Identificare pattern ricorrenti tra pazienti con malattie croniche.
- Fornire insight per migliorare la qualità del servizio e personalizzare le cure.
- Ridurre il carico sugli ospedali favorendo la deospedalizzazione.

## Struttura del Progetto e relative cartelle

### Data Preprocessing
La cartella **datapreprocessing** contiene file e sotto-cartelle per la preparazione dei dati utilizzati per l'analisi AI.

#### Sottocartella Manage
- **DataFix**: Corregge errori o incongruenze presenti nel dataset iniziale, garantendo l'integrità dei dati.
- **DataCleaner**: Pulisce il dataset eliminando duplicati, gestendo valori mancanti e assicurando una struttura dati coerente.
- **DataPlot**: Genera e salva visualizzazioni grafiche che rappresentano la distribuzione dei cluster e altri aspetti rilevanti dei dati.

#### Altri file in Datapreprocessing
- **ManageData**: Gestisce l'intero processo di manipolazione del dataset, coordinando le diverse fasi di preprocessamento come la pulizia, l'integrazione e la trasformazione dei dati.
- **FeatureSelection**: Identifica e seleziona le feature più rilevanti e utili per il modello, migliorando la performance e riducendo la complessità computazionale.
- **FeatureExtractor**: Estrae nuove feature significative dai dati grezzi, trasformandoli in una rappresentazione più utile per l'analisi e il training del modello.
- **Clustering**: Raggruppa i dati in insiemi o cluster di elementi simili, facilitando l'analisi delle strutture nascoste e dei modelli all'interno del dataset.

### Cartelle
- La cartella **Data** contiene i dataset e file di supporto per il progetto. Al suo interno, la sottocartella *italia* include file JSON con dati su comuni, regioni e province italiane, mentre i file *challenge_campus_biomedico_2024.paquet*, *extractor_dataset.parquet*, e *dataset_clustered.parquet* rappresentano rispettivamente il dataset principale, il dataset con feature estratte e il dataset clusterizzato.
- La cartella **graphs** contiene grafici che visualizzano i risultati dei test effettuati, mostrando le performance dei modelli di clustering e selezione delle feature.
- La cartella **saved_models** contiene un file pickle con un modello di clustering KMeans salvato.
- La cartella **results** Contiene un file Json per i risultati ottenuti es. Silhouette, purity ecc.

### Altri File
- **.gitignore**: Definisce i file e le cartelle da ignorare nel repository Git.
- **main.py**: Coordina tutte le operazioni, dal preprocessamento all'analisi dei dati.
- **Readme** 

### Gestione dei log del programma
**Logging**:
   - Tutto lo script utilizza il logging per segnalare eventuali errori, come file non trovati o problemi nella formattazione dei dati, e per indicare il completamento del processo.

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

1. **remove_duplicates(dataset)**:
   - Rimuove righe duplicate dal dataset.
   - Registra nel log quante righe duplicate sono state rimosse.

2. **remove_missing_values_rows(dataset, null_threshold=0.6)**:
   - Rimuove le colonne che hanno una percentuale di valori nulli superiore alla soglia specificata (default: 60%).
   - Registra nel log quali colonne sono state eliminate.

3. **remove_disdette(dataset)**:
   - Rimuove le righe in cui la colonna `data_disdetta` non è nulla (ossia le cancellazioni).
   - Registra nel log quante righe sono state rimosse.

4. **remove_columns(dataset, columns)**:
   - Elimina dal dataset le colonne specificate.
   - Registra nel log le colonne che sono state rimosse o notifica se nessuna colonna specificata è stata trovata.

5. **handle_missing_values(dataset, strategy='mean')**:
   - Gestisce i valori mancanti nel dataset, riempiendo i valori nulli in base alla strategia scelta (default: media, altre opzioni: 'median', 'mode').
   - Registra nel log il numero di valori mancanti trovati e gestiti, specificando la strategia utilizzata.

6. **update_dataset_with_outliers(dataset, relevant_columns=['eta_paziente', 'durata_visita', 'descrizione_attivita'], contamination=0.05, action='remove')**:
   - Identifica e gestisce gli outlier con un approccio ibrido che combina **Isolation Forest** e **Local Outlier Factor**.
   - Gli outlier possono essere rimossi o segnati, a seconda del parametro `action` (default: 'remove').
   - Registra nel log quanti outlier sono stati trovati e l'azione intrapresa.

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


## Clustering.py

### Parametri Principali
- **n_clusters**: Numero di cluster da utilizzare nel modello KModes.
- **use_one_hot**: Se impostato su True, utilizza One-Hot Encoding per le variabili categoriali; altrimenti utilizza Label Encoding.

### Metodi Principali
- **get_dataset_clustered()**: Restituisce il dataset con le etichette assegnate ai cluster.
  
- **elbow_method(self, dataset, min_clusters=2, max_clusters=6, threshold=0.05)**: Esegue l'Elbow Method per determinare il numero ottimale di cluster e salva il grafico nella directory graphs. Usa un'euristica basata su un threshold per identificare l'angolo (elbow) nel grafico delle distorsioni.

- **preprocess_data(self, dataset)**: Trasforma le variabili categoriali in numeriche usando One-Hot o Label Encoding, e standardizza i dati numerici per il clustering. Restituisce il dataset preprocessato e l'array dei cluster.

- **fit(self, dataset)**: Esegue il clustering con KModes, calcola il Silhouette Score e restituisce il dataset con le etichette assegnate ai cluster. Viene effettuata una verifica preliminare per rilevare valori nulli nel dataset.

- **calculate_purity(self, dataset, label_column='incremento_classificato')**: Calcola la purezza del clustering confrontando le etichette assegnate ai cluster con una colonna di riferimento (es. incremento_classificato). Supporta dataset trasformati con One-Hot Encoding.

- **plot_clusters(self, dataset)**: Crea e salva un grafico bidimensionale (2D) dei cluster usando la PCA per ridurre le dimensioni e un Silhouette plot. Entrambi i grafici vengono salvati nella directory graphs.

- **plot_clusters_3d(self, dataset)**: Crea e salva un grafico tridimensionale (3D) dei cluster, utilizzando la PCA per ridurre le dimensioni a 3 componenti principali. Il grafico viene salvato nella directory graphs.

- **run_clustering(self, dataset, label_column='incremento_classificato', excluded_columns=None)**: Esegue l'intero processo di clustering, che include il preprocessing dei dati, l'esecuzione del clustering, il calcolo della purezza, la creazione dei grafici e il salvataggio dei risultati finali.

### Output

- **Grafici Salvati**: Tutti i grafici generati durante il processo di clustering vengono salvati nella directory graphs:
  - Elbow Method (elbow_method.png)
  - PCA Plot 2D (pca_clusters.png)
  - PCA Plot 3D (pca_clusters_3d.png)
  - Silhouette Plot (silhouette_plot.png)

- **Risultati Clustering**: I risultati del clustering vengono salvati nella directory results in formato JSON:
  - clustering_results.json: Contiene informazioni sulle colonne escluse e utilizzate, il numero di cluster ottimali, il silhouette score medio, la purezza e la metrica finale calcolata.


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
   - Dataset ottimizzato senza variabili altamente correlate.
     
	 


### Conclusioni

Il progetto di **Clustering Supervisionato per la Teleassistenza** ha dimostrato come l'analisi dei dati e l'applicazione di tecniche di clustering possano migliorare significativamente la gestione e l'efficacia dei servizi di teleassistenza. Attraverso la profilazione dei pazienti basata su caratteristiche rilevanti, come età, durata delle visite, e distribuzione geografica, è possibile individuare pattern utili per personalizzare le cure e ridurre il carico sugli ospedali.

L'integrazione di strumenti avanzati come la selezione delle feature, l'estrazione delle caratteristiche, e la gestione dei dati, ha permesso di realizzare un workflow efficiente che può essere applicato a contesti reali di teleassistenza. La combinazione di algoritmi di clustering e tecniche di visualizzazione dei dati ha portato a una maggiore comprensione delle dinamiche del servizio, favorendo decisioni informate per migliorare la qualità dell'assistenza sanitaria a distanza.

I risultati ottenuti forniscono una base solida per ulteriori sviluppi, tra cui l'ottimizzazione del modello di clustering e l'implementazione di nuovi algoritmi per migliorare la precisione e la granularità dell'analisi. 


