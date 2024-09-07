
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



