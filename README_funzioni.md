
## ManageData.py

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
    


## DataFix.py

1. **`fetch_province_code_data(file_path)`**:
   - Carica i dati delle province da un file JSON.
   - Restituisce due dizionari: `codice_to_provincia` e `provincia_to_codice`, che mappano i codici delle province ai nomi e viceversa.
   - Registra nel log se il file non è trovato o se ci sono errori di decodifica.

2. **`fetch_comuni_code_data(file_path)`**:
   - Carica i dati dei comuni da un file JSON.
   - Restituisce due dizionari: `codice_to_comune` e `comune_to_codice`, che mappano i codici dei comuni ai nomi e viceversa.
   - Registra nel log se il file non è trovato o se ci sono errori di decodifica.

3. **`process_province_comuni(dataset, codice_to_provincia, provincia_to_codice, codice_to_comune, comune_to_codice)`**:
   - Processa il dataset per riempire i campi relativi ai comuni e alle province utilizzando i dizionari di mappatura.
   - Registra nel log se ci sono colonne mancanti nel dataset.

4. **`fill_province_comuni(dataset, path_province, path_comuni)`**:
   - Riempie i dati mancanti relativi alle province e ai comuni nel dataset utilizzando i file JSON forniti.
   - Registra nel log eventuali errori di caricamento dei dati.

5. **`add_durata_visita(dataset)`**:
   - Calcola la durata della visita per ciascuna riga, aggiungendo una colonna `durata_visita` che rappresenta la durata in minuti.
   - Gestisce casi in cui le ore di inizio o fine erogazione non sono valide o mancanti, e registra tali righe nel log.

6. **`add_eta_paziente(dataset)`**:
   - Calcola l'età del paziente basata sulla data di nascita e aggiunge una colonna `eta_paziente`.
   - Rimuove valori non plausibili di età e registra il numero di valori invalidi nel log.

7. **`fill_durata_visita(dataset)`**:
   - Riempie le durate di visita mancanti utilizzando la media delle durate per ciascun tipo di servizio.
   - Registra nel log quante righe hanno ancora durate mancanti.

8. **`add_fascia_eta_column(dataset)`**:
   - Genera una nuova colonna `fascia_eta` basata sull'età del paziente.
   - Normalizza la colonna `eta_paziente` tra 0 e 1 utilizzando MinMaxScaler.

9. **`colonne_to_category(df, colonne)`**:
   - Converte le colonne specificate in 'category' per ottimizzare la memoria.
   - Se possibile, converte anche la colonna `codice_struttura_erogazione` in `int64`.


## Datacleaner.py

1. **`remove_duplicates(dataset)`**:
   - Rimuove righe duplicate dal dataset.
   - Registra nel log quante righe duplicate sono state rimosse.

2. **`remove_missing_values_rows(dataset, null_threshold=0.6)`**:
   - Rimuove le colonne che hanno una percentuale di valori nulli superiore alla soglia specificata (default: 60%).
   - Registra nel log quali colonne sono state eliminate.

3. **`remove_disdette(dataset)`**:
   - Rimuove le righe in cui la colonna data_disdetta non è nulla (ossia le cancellazioni).
   - Registra nel log quante righe sono state rimosse.

4. **`remove_columns(dataset, columns)`**:
   - Elimina dal dataset le colonne specificate.
   - Registra nel log le colonne che sono state rimosse o notifica se nessuna colonna specificata è stata trovata.

5. **`handle_missing_values(dataset, strategy='mean')`**:
   - Gestisce i valori mancanti nel dataset, riempiendo i valori nulli in base alla strategia scelta (default: media, altre opzioni: 'median', 'mode').
   - Registra nel log il numero di valori mancanti trovati e gestiti, specificando la strategia utilizzata.

6. **`update_dataset_with_outliers(dataset, relevant_columns=['eta_paziente', 'durata_visita', 'descrizione_attivita'], contamination=0.05, action='remove')`**:
   - Identifica e gestisce gli outlier con un approccio ibrido che combina *Isolation Forest* e *Local Outlier Factor*.
   - Gli outlier possono essere rimossi o segnati, a seconda del parametro action (default: 'remove').
   - Registra nel log quanti outlier sono stati trovati e l'azione intrapresa.


## FeatureSelection.py

1. **`__init__(self, df, categorical_columns=None, exclude_classes=None)`**:
   - Inizializza l'oggetto FeatureSelection con il DataFrame e le colonne categoriali.
   - Registra nel log le colonne categoriali rilevate automaticamente o specificate dall'utente.
   - Esclude eventuali colonne specificate dall'utente.

2. **`get_dataset(self)`**:
   - Restituisce il DataFrame attualmente in uso.

3. **`get_colum_to_drop(self)`**:
   - Restituisce l'elenco delle colonne rimosse durante il processo di selezione delle caratteristiche.

4. **`calculate_cramers_v(self, column1, column2)`**:
   - Calcola il valore di Cramér's V per misurare la correlazione tra due colonne categoriali.
   - Restituisce il valore di Cramér's V.

5. **`create_correlation_matrix(self)`**:
   - Crea una matrice di correlazione tra le colonne categoriali utilizzando Cramér's V.
   - Restituisce la matrice di correlazione come DataFrame.

6. **`remove_perfectly_correlated_features(self, corr_matrix, threshold=1.0)`**:
   - Rimuove colonne con correlazione perfetta (Cramér's V pari a 1) dal DataFrame.
   - Registra nel log le colonne rimosse e quelle rimanenti.

7. **`remove_highly_correlated_features(self, corr_matrix, threshold=0.8)`**:
   - Rimuove colonne con alta correlazione (Cramér's V superiore alla soglia specificata) dal DataFrame.
   - Registra nel log le colonne rimosse e quelle rimanenti.

8. **`display_heatmap(self, corr_matrix, title, filename)`**:
   - Visualizza e salva una heatmap basata sulla matrice di correlazione.
   - Salva la heatmap come immagine in una cartella chiamata 'graphs'.

9. **`execute_feature_selection(self, threshold=0.8, remove_others_colum_by_threshold=False)`**:
   - Esegue l'intero processo di selezione delle caratteristiche.
   - Rimuove colonne perfettamente e altamente correlate, e salva le heatmap iniziale e finale.



## FeatureExtractor.py

1. **`__init__(self, dataset)`**:
   - Inizializza la classe FeatureExtractor con il dataset fornito.
   - Registra l'inizio dell'estrazione delle caratteristiche.

2. **`get_dataset(self)`**:
   - Restituisce il dataset attualmente in uso.

3. **`preprocess_data(self)`**:
   - Preprocessa i dati per l'analisi:
     - Converte la colonna `data_erogazione` in formato datetime.
     - Crea colonne aggiuntive per anno e trimestre.
   - Restituisce il dataset preprocessato.

4. **`calculate_percentage_increments(self)`**:
   - Calcola gli incrementi percentuali del numero di servizi per trimestre e codice descrizione attività.
   - Restituisce il dataset con gli incrementi calcolati.

5. **`determine_growth_category(self, variazione_percentuale)`**:
   - Determina la categoria di crescita (decrescita, crescita costante, bassa, moderata, alta) in base alla variazione percentuale.

6. **`apply_growth_categorization(self, grouped)`**:
   - Applica la categorizzazione della crescita percentuale e unisce i risultati al dataset originale.

7. **`plot_graphs(self, grouped)`**:
   - Crea e salva grafici della distribuzione degli incrementi percentuali e della crescita trimestrale dei servizi di teleassistenza.
   - Salva i grafici nella cartella `graphs`.

8. **`run_analysis(self)`**:
   - Esegue l'intera pipeline di analisi:
     - Preprocessing dei dati.
     - Calcolo degli incrementi percentuali.
     - Classificazione degli incrementi.
     - Generazione dei grafici.
    
## Clustering.py

1. **`__init__(self, n_clusters=4, use_one_hot=False)`**:
   - Inizializza la classe per eseguire il clustering con KModes.
   - Registra nel log l'inizializzazione del clustering.
   - Parametri: 
     - `n_clusters`: numero di cluster (default=4).
     - `use_one_hot`: se True, utilizza One-Hot Encoding; altrimenti usa Label Encoding.

2. **`get_dataset_clustered(self)`**:
   - Restituisce il dataset con le etichette del cluster.

3. **`get_dataset_with_cluster(self)`**:
   - Restituisce il dataset con le colonne e le etichette del cluster.

4. **`check_null_values(self, dataset)`**:
   - Verifica la presenza di righe con valori nulli nel dataset.

5. **`stratified_downsample(self, dataset, target_column, fraction=0.3)`**:
   - Riduce il dataset mantenendo la proporzione delle classi nel target (default: 30%).

6. **`elbow_method(self, dataset, min_clusters=2, max_clusters=10, threshold=0.05)`**:
   - Esegue l'Elbow Method per determinare il numero ottimale di cluster.
   - Salva il grafico del metodo Elbow nella cartella `graphs`.

7. **`preprocess_data(self, dataset)`**:
   - Preprocessa le colonne categoriali utilizzando One-Hot o Label Encoding.
   - Standardizza il dataset preprocessato e restituisce i dati trasformati.

8. **`fit(self, dataset)`**:
   - Esegue il clustering KModes sul dataset preprocessato.
   - Restituisce il dataset con le etichette di cluster aggiunte.

9. **`calculate_final_metric(self)`**:
   - Calcola la metrica finale combinando purezza e silhouette score con una penalizzazione per il numero di cluster.

10. **`calculate_purity(self, dataset, label_column='incremento_classificato')`**:
    - Calcola la purezza del clustering in base alla colonna `incremento_classificato`.

11. **`plot_clusters(self, dataset, cluster_column='cluster')`**:
    - Esegue il plot dei cluster in 2D utilizzando PCA e Silhouette Score.
    - Salva i plot nella cartella `graphs`.

12. **`plot_clusters_3d(self, dataset, cluster_column='cluster')`**:
    - Esegue il plot dei cluster in 3D utilizzando PCA.
    - Salva i grafici nella cartella `graphs`.

13. **`save_results(self, excluded_columns, used_columns)`**:
    - Salva i risultati del clustering in un file JSON nella cartella `results`.

14. **`run_clustering(self, dataset, label_column='incremento_classificato', excluded_columns=None)`**:
    - Esegue l'intero processo di clustering, inclusi KModes, silhouette score, purezza e plotting.
    - Salva i risultati finali.



## DataPlot.py

1. **`__init__(self, df)`**:
   - Inizializza la classe DataPlot con il dataset fornito e crea la cartella 'graphs' se non esiste.

2. **`save_plot(self, plt, filename)`**:
   - Salva il grafico nella cartella 'graphs'.

3. **`ensure_correct_column_types(self)`**:
   - Assicura che le colonne categoriali e numeriche siano del tipo corretto.

4. **`plot_cluster_distribution(self)`**:
   - Crea il grafico della distribuzione dei cluster.

5. **`plot_sex_distribution_by_cluster(self)`**:
   - Crea il grafico della distribuzione del sesso per cluster.

6. **`plot_age_distribution_by_cluster(self)`**:
   - Crea il grafico della distribuzione delle fasce d'età per cluster.

7. **`plot_age_by_region(self)`**:
   - Mostra la distribuzione delle visite per fascia d'età in base alla regione.

8. **`plot_cluster_by_region(self)`**:
   - Mostra la distribuzione dei cluster per regione.

9. **`plot_cluster_by_structure_type(self)`**:
   - Crea il grafico della distribuzione dei cluster per tipologia di struttura.

10. **`plot_increment_by_cluster(self)`**:
    - Crea il grafico della variabile `incremento_classificato` per cluster.

11. **`plot_cluster_by_quarter(self)`**:
    - Crea il grafico della distribuzione dei cluster per trimestre.

12. **`plot_cluster_by_year(self)`**:
    - Crea il grafico della distribuzione dei cluster per anno.

13. **`plot_cluster_by_professional(self, n=10)`**:
    - Crea il grafico della distribuzione dei cluster per i primi `n` professionisti sanitari più presenti.

14. **`plot_visit_duration_by_cluster(self)`**:
    - Mostra la durata delle visite per cluster.

15. **`plot_visit_duration_strip_by_cluster(self)`**:
    - Mostra la durata delle visite per cluster (strip plot).

16. **`plot_visit_duration_by_sex(self)`**:
    - Mostra la durata delle visite per sesso.

17. **`plot_visit_duration_by_region_and_sex(self)`**:
    - Mostra la durata delle visite per regione e sesso.

18. **`plot_visits_by_age_and_sex(self)`**:
    - Mostra il numero di visite per fascia d'età e sesso.

19. **`plot_visits_by_year_and_region(self)`**:
    - Mostra la distribuzione delle visite per anno e regione.

20. **`plot_visit_duration_by_age_and_cluster(self)`**:
    - Mostra la durata delle visite per fascia d'età e cluster.

21. **`plot_total_professionals_per_region(self)`**:
    - Visualizza il numero di professionisti sanitari unici per regione.

22. **`plot_professionals_by_type_and_region(self)`**:
    - Visualizza il numero di professionisti sanitari per tipologia in ogni regione.

23. **`generate_plots(self)`**:
    - Esegue tutti i metodi per generare i grafici e salvarli nella cartella 'graphs'.
