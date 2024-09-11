
# Progetto: Clustering Supervisionato per la Teleassistenza

Questo progetto mira a profilare i pazienti in base al loro utilizzo del servizio di Teleassistenza, utilizzando tecniche di clustering supervisionato. L'obiettivo principale è identificare gruppi di pazienti con comportamenti simili, in funzione dell'incremento delle teleassistenze fornite. Attraverso l'analisi dei cluster, si possono comprendere i fattori che influenzano l'aumento dell'uso del servizio, migliorando così la gestione e l'erogazione delle cure a distanza.


##Obiettivi
L'obiettivo principale del progetto è migliorare la comprensione dei fattori che determinano l'aumento dell'uso del servizio di Teleassistenza, specialmente tra i pazienti con malattie croniche. Questo verrà realizzato creando modelli di clustering supervisionato per identificare gruppi di pazienti con comportamenti simili. 
Gli obiettivi specifici includono: 
1) individuare i pattern ricorrenti che portano a un maggiore ricorso alla Teleassistenza.
2) fornire insight per migliorare la qualità e l'efficienza del servizio.
3) aiutare a personalizzare le cure e a ridurre il carico sugli ospedali, favorendo così la deospedalizzazione.


##Suddivisione
Nella repository *https://github.com/Lele991/Project_Telemedicine_FIA24.git* sono presenti delle cartelle che contengono i codici e le logiche attraverso le quali si è gestito il problema.
Vi è:

# Data
La cartella **data** contiene una sottocartella **Italia**.

## Sottocartella
Contiene JSON con le informazioni di tutti i comuni,regioni e province italiane, dati mediate i quali vengono successivamente implementate delle logiche.

## File Principali
Traccia del progetto.


# Data Preprocessing

La cartella **datapreprocessing** contiene vari file e una sottocartella per la preparazione dei dati, necessari per l'analisi AI.

## Sottocartella
- **datafix**: Corregge eventuali errori o incongruenze nei dati.
- **datacleaner**: Si occupa della pulizia dei dati, rimuovendo valori mancanti, duplicati o non rilevanti.

## File Principali
- **clustering**: Organizza i dati in gruppi simili (cluster).
- **featureselection**: Seleziona le feature più rilevanti per il modello.
- **featureextractor**: Estrae nuove feature dai dati grezzi.
- **managedata**: Gestisce i dati complessivi, integrando varie fasi di preprocessamento.


# Graph

La cartella **graph** contiene i risultati dei test effettuati sotto forma di grafici, che visualizzano l'andamento e le performance dei vari modelli di clustering e feature selection.

# Altri File

- **.gitignore**: Definisce i file e le cartelle da ignorare nel repository Git, per evitare di caricare file temporanei o di sistema.
- **main**: È il file principale che esegue il workflow completo, orchestrando le operazioni di preprocessamento e analisi dei dati.

Questi componenti lavorano insieme per ottimizzare i dati, analizzarli e visualizzare i risultati ottenuti dai vari modelli AI.

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

## Log delle Operazioni

Il modulo utilizza il logging per tracciare ogni fase del processo di selezione delle caratteristiche, registrando messaggi informativi, avvisi e risultati delle operazioni.

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

  ## Log delle Operazioni

  Il modulo utilizza il logging per tenere traccia di ogni fase dell'analisi. Viene creato un log dettagliato di ogni fase, che include informazioni sull'inizio e il completamento dei diversi passaggi, oltre a eventuali warning per dati mancanti o non validi.

  ## Output

   - Heatmap delle correlazioni (iniziale e finale) salvate in graphs.
   -  Dataset ottimizzato senza variabili altamente correlate.
     
	 





