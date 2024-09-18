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
---
## ManageData.py
Questo file gestisce l'intero flusso di preprocessing, analisi e clustering dei dati relativi ai servizi di teleassistenza, assicurando che il dataset sia pulito, preparato e organizzato per l'analisi e il clustering.
#### Funzionalità principali:
- Sostituisce valori mancanti e standardizza i dati per migliorare la qualità del dataset.
- Logga informazioni dettagliate sui valori mancanti e sulle colonne del dataset per monitorare la pulizia dei dati.
- Salva il dataset processato in formato Parquet, garantendo persistenza dei dati preprocessati.
- Esegue operazioni di pulizia completa dei dati, rimuovendo duplicati, cancellazioni e gestendo valori nulli.
- Analizza i dati, calcola metriche come la durata della visita e l'età del paziente, e individua gli outlier.
- Seleziona e crea le feature più rilevanti attraverso processi di **FeatureSelection** e **FeatureExtractor**.
- Applica il clustering per organizzare i dati in gruppi e aggiunge etichette di cluster al dataset.
- Genera grafici per visualizzare i risultati dell'analisi e del clustering, facilitando l'interpretazione dei dati.

## DataFix.py
Questo file gestisce la correzione e l'arricchimento dei dati all'interno del dataset. Fornisce funzionalità per completare campi mancanti, correggere valori errati, e aggiungere nuove colonne derivate dai dati esistenti.
#### Funzionalità principali:
- Carica e utilizza dati esterni per mappare codici a nomi di province e comuni, facilitando la correzione di campi relativi alla localizzazione.
- Aggiunge colonne calcolate come la durata della visita e l'età del paziente, migliorando la completezza e qualità dei dati.
- Gestisce i valori mancanti nella durata delle visite utilizzando dati disponibili o calcolando una durata media.
- Introduce una categorizzazione delle età in fasce, standardizzando i dati dei pazienti.
- Converte colonne in tipi categorici per ottimizzare l'uso della memoria nel dataset.

## DataCleaner.py
Questo file è responsabile della pulizia del dataset, assicurando che i dati siano privi di duplicati, valori mancanti e outlier. Fornisce varie funzioni per gestire dati inconsistenti e prepararli per le fasi successive di analisi e modellazione.
#### Funzionalità principali:
- Rimuove righe duplicate e registra nel log il numero di duplicati eliminati.
- Elimina le colonne con troppi valori mancanti, rispettando una soglia configurabile.
- Filtra le righe corrispondenti a cancellazioni, come indicato nella colonna `data_disdetta`.
- Permette la rimozione di colonne specifiche dal dataset, segnalando eventuali eliminazioni non valide.
- Gestisce i valori mancanti attraverso diverse strategie (media, mediana o moda) e riempie i valori nulli.
- Identifica e gestisce gli outlier utilizzando algoritmi avanzati come *Isolation Forest* e *Local Outlier Factor*, con opzioni per rimuoverli o segnarli nel dataset.

## FeatureSelection.py
La classe `FeatureSelection` è progettata per eseguire la selezione delle caratteristiche categoriali in un dataset. Utilizza la V di Cramér per calcolare la correlazione tra le variabili categoriali e rimuovere le caratteristiche altamente o perfettamente correlate. Fornisce una pipeline completa per gestire la selezione delle feature in un unico passaggio, con visualizzazioni delle correlazioni tramite heatmap.
#### Funzionalità principali
- **Calcolo della correlazione categoriale**: Calcola la correlazione tra variabili categoriali utilizzando la V di Cramér.
- **Rimozione delle caratteristiche correlate**: Rimuove le colonne perfettamente correlate o con correlazione superiore a una soglia definita dall'utente.
- **Generazione di heatmap**: Visualizza le correlazioni tra le variabili prima e dopo la selezione delle caratteristiche, salvando i grafici in formato immagine.
- **Esecuzione automatizzata**: Fornisce una pipeline che esegue l'intero processo di selezione delle feature con un singolo comando.
**Output**:
- Le heatmap di correlazione sono salvate nella directory `graphs` con il nome `combined_plot.png`, mostrando la correlazione delle feature prima e dopo il processo di selezione.

## FeatureExtraction.py
La classe `FeatureExtractor` esegue un'analisi del dataset per calcolare gli incrementi percentuali dei servizi erogati nel tempo, categorizza le variazioni di crescita e crea grafici per visualizzare l'andamento dei dati. Fornisce una pipeline completa che include il preprocessamento, il calcolo degli incrementi e la visualizzazione delle tendenze dei servizi di teleassistenza su base trimestrale e annuale.
#### Funzionalità principali
- **Preprocessamento dei dati**: Converte le date in formati adatti per l'analisi e crea colonne aggiuntive per anno e trimestre.
- **Calcolo degli incrementi percentuali**: Raggruppa i dati per anno, trimestre e attività per calcolare le variazioni percentuali dei servizi rispetto ai periodi precedenti.
- **Categorizzazione della crescita**: Classifica le variazioni percentuali in categorie come 'crescita costante' o 'decrescita'.
- **Visualizzazione grafica**: Genera grafici, come istogrammi e boxplot, per visualizzare la distribuzione degli incrementi e l'andamento trimestrale.
- **Esecuzione automatizzata**: Esegue l'intera analisi in sequenza, dal preprocessamento alla visualizzazione.
#### Output
- I grafici di correlazione e distribuzione degli incrementi percentuali sono salvati nella cartella `graphs`.
- Il dataset con le categorie di crescita aggiunte e ottimizzato per l'analisi.

## Clustering.py
Questo file gestisce l'intero processo di clustering, dalla preparazione dei dati alla scelta del numero ottimale di cluster, fino alla valutazione del modello e alla generazione di grafici.
#### Funzionalità principali:
- **Determinazione del numero ottimale di cluster**: Utilizza l'Elbow Method per trovare il numero ottimale di cluster da utilizzare nel modello.
- **Preprocessing dei dati**: Converte le variabili categoriali in numeriche e standardizza i dati per prepararli al clustering.
- **Esecuzione del clustering**: Esegue il clustering utilizzando KModes e calcola il Silhouette Score, la purity e la metrica finale per valutare la qualità del clustering.
- **Valutazione della purezza**: Confronta i cluster con una colonna di riferimento per calcolare la purezza del modello.
- **Visualizzazione dei cluster**: Genera grafici bidimensionali e tridimensionali dei cluster usando la PCA, insieme a un Silhouette plot.
- **Esportazione dei risultati**: Salva i grafici e i risultati finali del clustering in formato JSON per l'analisi successiva.
#### Output
- Grafici salvati nella directory `graphs`: 
  - Elbow Method, PCA 2D, PCA 3D, Silhouette plot.
- Risultati salvati in `results`: 
  - `clustering_results.json` con dettagli sul numero di cluster, purezza e performance del modello.

## DataPlot.py
Questo file si occupa della generazione e salvataggio di grafici che visualizzano diverse distribuzioni e relazioni nel dataset, con particolare attenzione alla rappresentazione grafica dei cluster.
#### Funzionalità principali:
- **Generazione di grafici per la distribuzione dei dati**: Fornisce vari tipi di grafici per analizzare la distribuzione dei cluster, del sesso, delle fasce di età e della durata delle visite.
- **Analisi regionale e strutturale**: Crea grafici che mostrano come i cluster sono distribuiti nelle regioni di residenza e nelle diverse tipologie di strutture.
- **Incrementi classificati e temporalità**: Visualizza gli incrementi per cluster e la loro distribuzione nel tempo, suddivisi per trimestre.
- **Esportazione dei grafici**: Salva tutti i grafici generati nella directory dedicata 'graphs' per facilitare l'analisi e la consultazione visiva.


## Conclusioni
Il progetto di **Clustering Supervisionato per la Teleassistenza** ha dimostrato come l'analisi dei dati e l'applicazione di tecniche di clustering possano migliorare significativamente la gestione e l'efficacia dei servizi di teleassistenza. Attraverso la profilazione dei pazienti basata su caratteristiche rilevanti, come età, durata delle visite, e distribuzione geografica, è possibile individuare pattern utili per personalizzare le cure e ridurre il carico sugli ospedali.

L'integrazione di strumenti avanzati come la selezione delle feature, l'estrazione delle caratteristiche, e la gestione dei dati, ha permesso di realizzare un workflow efficiente che può essere applicato a contesti reali di teleassistenza. La combinazione di algoritmi di clustering e tecniche di visualizzazione dei dati ha portato a una maggiore comprensione delle dinamiche del servizio, favorendo decisioni informate per migliorare la qualità dell'assistenza sanitaria a distanza.

I risultati ottenuti forniscono una base solida per ulteriori sviluppi, tra cui l'ottimizzazione del modello di clustering e l'implementazione di nuovi algoritmi per migliorare la precisione e la granularità dell'analisi. 

# Utilizzo
<details>
<summary>Mostra tutto</summary>
   
1. **Installazione**:
   - Clona il repository:

     ```bash
     git clone https://github.com/Lele991/Project_Telemedicine_FIA24
     cd Project_Telemedicine_FIA24
     ```

2. **Creazione dell'ambiente virtuale** (non necessario):
   - Crea un ambiente virtuale e lo attiva:

     ```bash
     python3 -m venv venv
     source venv/bin/activate  # Su Windows: venv\Scripts\activate
     ```

3. **Installazione dei requisiti**:
   - Installa le dipendenze richieste:

     ```bash
     pip install -r requirements.txt
     ```

4. **Esecuzione del progetto**:
   - Per avviare l'analisi e l'elaborazione del dataset, esegui il file `main.py`:

     ```bash
     python main.py
     ```

Il file `main.py` avvierà il processo di preprocessing, analisi, e clustering dei dati.
</details>
