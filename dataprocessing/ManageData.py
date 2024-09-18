import numpy as np
import pandas as pd
import logging

from dataprocessing.Clustering import Clustering
from dataprocessing.FeatureSelection import FeatureSelection
from dataprocessing.FeatureExtractor import FeatureExtractor
from dataprocessing.manage import DataCleaner, DataFix
from dataprocessing.manage.DataPlot import DataPlot

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ManageData:
    def __init__(self, dataset, path_province, path_comuni, missing_threshold=0.6):
        self.dataset = dataset
        self.path_province = path_province
        self.path_comuni = path_comuni
        self.missing_threshold = missing_threshold

    def get_dataset(self):
        return self.dataset
    
    def set_dataset(self, dataset):
        self.dataset = dataset

    def replace_none_with_nan(self, dataset):
        """
        Sostituisce i valori 'None' e None con NaN nel DataFrame.
        
        Parametri:
        dataset (pd.DataFrame): Il DataFrame in cui eseguire la sostituzione.
        
        Ritorna:
        dataset: Il DataFrame con 'None' e None sostituiti con NaN.
        """
        # Verifica che il dataset sia un DataFrame
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError("Il parametro deve essere un DataFrame di pandas.")
        
        # Sostituisce 'None' e None con NaN
        dataset.replace({'None': np.nan, None: np.nan}, inplace=True)
        logging.info("Sostituiti i valori 'None' e None con NaN nel dataset.")

    def log_missing_values(self, dataset):
        """Stampa un log con le colonne che contengono valori mancanti con annesso valore."""
        missing_values = dataset.isnull().sum()
        total_missing = missing_values.sum()
        if total_missing > 0:
            logging.info(f"Valori mancanti trovati in totale: {total_missing}")
            missing_by_column = missing_values[missing_values > 0]
            for col, missing_count in missing_by_column.items():
                logging.info(f"La colonna '{col}' ha {missing_count} valori mancanti.")

    def save_dataset(self, dataset, name='extractor_dataset'):
        file_path = 'data/' + name + '.parquet'
        """Salva il dataset in formato Parquet."""
        dataset.to_parquet(file_path, index=False)
        logging.info(f"Dataset salvato in formato Parquet: {file_path}")

    def print_columns(self, dataset):
        """Stampa le colonne del dataset."""
        logging.info(f"Colonne presenti nel dataset: {dataset.columns.tolist()}")

    def clean_data(self):
        """Esegue una pulizia completa dei dati."""
        logging.info("Inizio della pulizia completa del dataset.")

        df = self.dataset.copy()

        # Sostituisco i valori None(nulli) con "NaN" nel Dataframe
        self.replace_none_with_nan(df)

        # Rimuove righe dove 'data_disdetta' non è nullo
        df = DataCleaner.remove_disdette(df)
        
        # Fix province e comuni
        df = DataFix.fill_province_comuni(df, self.path_province, self.path_comuni)

        # Rimuove duplicati
        df = DataCleaner.remove_duplicates(df)

        # Rimuove colonne non più necessarie
        columns = ['data_disdetta']
        df = DataCleaner.remove_columns(df, columns)

        # Rimuove colonne con valori mancanti sopra la soglia
        df = DataCleaner.remove_missing_values_rows(df, self.missing_threshold)

        logging.info("Fine della pulizia dei dati.")
        return df
    
    def run_analysis(self):
        """Esegue l'analisi completa dei dati."""
        logging.info("Inizio dell'analisi completa dei dati.")
        df = self.clean_data()

        # Aggiunge durata della visita e riempie le durate mancanti
        df = DataFix.add_durata_visita(df)
        df = DataFix.fill_durata_visita(df)
        
        # Aggiunge l'età del paziente
        df = DataFix.add_eta_paziente(df)

        # Aggiunge la fascia d'età
        df = DataFix.add_fascia_eta_column(df)

        # Identifica e gestisce gli outlier
        relevant_columns = ['eta_paziente', 'durata_visita', 'codice_descrizione_attivita'] # Subset di colonne rilevanti
        df = DataCleaner.update_dataset_with_outliers(df, relevant_columns)

        print(df.info())

        # Rimuove colonne non più necessarie
        #columns = ['ora_inizio_erogazione', 'ora_fine_erogazione',
        #           'id_prenotazione', 'id_paziente', 'codice_regione_residenza', 'codice_regione_residenza',
        #           'asl_residenza', 'codice_asl_residenza', 'provincia_residenza',
        #           'comune_residenza', 'codice_comune_residenza', 'codice_descrizione_attivita',
        #           'data_contatto', 'regione_erogazione', 'asl_erogazione', 'provincia_erogazione',
        #           'codice_tipologia_struttura_erogazione', 'id_professionista_sanitario',
        #           'codice_tipologia_professionista_sanitario', 'codice_tipologia_professionista_sanitario']

        columns = ['id_paziente', 'tipologia_servizio', 'ora_inizio_erogazione', 'ora_fine_erogazione','data_contatto',
                   'codice_regione_residenza', 'provincia_residenza', 'comune_residenza',
                   'codice_descrizione_attivita', 'regione_erogazione', 'asl_erogazione', 'provincia_erogazione',
                   'codice_tipologia_professionista_sanitario',
                   'struttura_erogazione', 'codice_tipologia_struttura_erogazione',
                   'asl_residenza', 'outlier', 'eta_paziente']
        df = DataCleaner.remove_columns(df, columns)

        # Gestisce i valori mancanti nel dataset, riempiendo i valori mancanti in base alla media
        df = DataCleaner.handle_missing_values(df)

        # Imposta il dataset pulito
        logging.info("Pulizia completa del dataset eseguita con successo.")

        self.print_columns(df)
        print(df.info())
        print(df)

        # Colonne presenti nel dataset:
        # ['id_prenotazione', 'sesso', 'codice_regione_residenza',
        #  'codice_asl_residenza', 'codice_provincia_residenza',
        #  'codice_comune_residenza', 'codice_descrizione_attivita',
        #  'codice_regione_erogazione', 'codice_asl_erogazione', 'codice_provincia_erogazione',
        #  'codice_struttura_erogazione', 'codice_codice_tipologia_struttura_erogazione',
        #  'id_professionista_sanitario', 'codice_tipologia_professionista_sanitario',
        #  'data_erogazione', 'durata_visita', 'fascia_eta']

        columns = ['id_prenotazione', 'data_erogazione', 'id_professionista_sanitario']
        featureSelection = FeatureSelection(df, categorical_columns=None, exclude_classes=columns)
        featureSelection.execute_feature_selection(threshold=0.85, remove_others_colum_by_threshold=True)
        featureSelection.get_dataset()
        get_column_to_drop = featureSelection.get_colum_to_drop()
        
        self.print_columns(df)
        print(df.info())
        print(df)

        # Colonne tipicamente rimosse:
        # ['codice_provincia_erogazione', 'codice_provincia_residenza', 'codice_asl_erogazione',
        #  'codice_asl_residenza', 'codice_struttura_erogazione', 'id_professionista_sanitario',
        #  'codice_comune_residenza', 'codice_regione_erogazione', 'codice_tipologia_professionista_sanitario']
        df = DataCleaner.remove_columns(df, get_column_to_drop)

        self.print_columns(df)
        print(df.info())
        print(df)

        # Colonne presenti nel dataset
        # ['id_prenotazione', 'sesso', 'codice_regione_residenza',
        #  'codice_descrizione_attivita', 'codice_codice_tipologia_struttura_erogazione',
        #  'id_professionista_sanitario', 'codice_tipologia_professionista_sanitario',
        #  'data_erogazione', 'durata_visita', 'fascia_eta']

        featureExtractor = FeatureExtractor(df)
        featureExtractor.run_analysis()
        df = featureExtractor.get_dataset()

        self.print_columns(df)
        print(df.info())
        print(df)

        self.log_missing_values(df)
        self.save_dataset(df)

        # Colonne presenti nel dataset
        # ['id_prenotazione', 'sesso', 'codice_regione_residenza',
        #  'codice_descrizione_attivita', 'codice_codice_tipologia_struttura_erogazione',
        #  'id_professionista_sanitario', 'codice_tipologia_professionista_sanitario',
        #  'data_erogazione', 'durata_visita', 'fascia_eta', 'trimestre',
        #  'anno', 'incremento_classificato']        

        # columns_to_remove = ['id_prenotazione', 'data_erogazione', 'trimestre', 'id_professionista_sanitario']
        columns_to_remove = ['id_prenotazione', 'data_erogazione', 'id_professionista_sanitario', 'durata_visita']
        use_one_hot_encoding = False  # Se vuoi usare One-Hot Encoding
        # Creazione e esecuzione del clustering
        clustering = Clustering(n_clusters=4, use_one_hot=use_one_hot_encoding)
        clustering.run_clustering(df, label_column='incremento_classificato', excluded_columns=columns_to_remove)
        dataset_clustered = clustering.get_dataset_with_cluster()

        # Aggiungi la colonna 'cluster' dal dataset_clustered al dataset df
        df['cluster'] = dataset_clustered['cluster']
        self.save_dataset(df, name='dataset_clustered')
        
        # Inizializza il costruttore DataPlot
        data_plot = DataPlot(df)
        data_plot.generate_plots()


'''
Prove effettuate:

1)
columns_to_remove = ['id_prenotazione', 'data_erogazione', 'id_professionista_sanitario', 'durata_visita']
use_one_hot_encoding = True
Clustering KModes completato. Numero di Cluster: 4, Silhouette Score Medio: 0.7097433217724045, Purezza: 0.7245868316394167

2)
columns_to_remove = ['id_prenotazione', 'data_erogazione', 'id_professionista_sanitario', 'durata_visita']
use_one_hot_encoding = False
Clustering KModes completato. Numero di Cluster: 4, Silhouette Score Medio: 0.6973225099550656., Purezza: 0.5667719840919134

3)
attivando anche il rimuovi outliers nel preprocess_data
columns_to_remove = ['id_prenotazione', 'data_erogazione', 'id_professionista_sanitario', 'durata_visita']
use_one_hot_encoding = False


'''