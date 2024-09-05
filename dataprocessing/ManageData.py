import numpy as np
import pandas as pd
import logging

from dataprocessing.Clustering import Clustering
from dataprocessing.FeatureSelection import FeatureSelection
from dataprocessing.FeatureExtractor import FeatureExtractor
from dataprocessing.manage import DataCleaner, DataFix

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ManageData:
    '''
    ggjgjgjg
    hjhhhj
    '''
    def __init__(self, dataset, path_province, path_comuni, missing_threshold=0.6):
        self.dataset = dataset
        self.path_province = path_province
        self.path_comuni = path_comuni
        self.missing_threshold = missing_threshold

    def get_dataset(self):
        return self.dataset
    
    def set_dataset(self, dataset):
        self.dataset = dataset

    def replace_none_with_nan(self):
        """
        Sostituisce i valori 'None' e None con NaN nel DataFrame.
        
        Parametri:
        dataset (pd.DataFrame): Il DataFrame in cui eseguire la sostituzione.
        
        Ritorna:
        dataset: Il DataFrame con 'None' e None sostituiti con NaN.
        """
        # Verifica che il dataset sia un DataFrame
        if not isinstance(self.dataset, pd.DataFrame):
            raise TypeError("Il parametro deve essere un DataFrame di pandas.")
        
        # Sostituisce 'None' e None con NaN
        self.dataset.replace({'None': np.nan, None: np.nan}, inplace=True)
        logging.info("Sostituiti i valori 'None' e None con NaN nel dataset.")

    def log_missing_values(self):
        """Stampa un log con le colonne che contengono valori mancanti con annesso valore."""
        missing_values = self.dataset.isnull().sum()
        total_missing = missing_values.sum()
        if total_missing > 0:
            logging.info(f"Valori mancanti trovati in totale: {total_missing}")
            missing_by_column = missing_values[missing_values > 0]
            for col, missing_count in missing_by_column.items():
                logging.info(f"La colonna '{col}' ha {missing_count} valori mancanti.")

    def save_dataset(self):
        file_path = 'data/extractor_dataset.parquet'
        """Salva il dataset in formato Parquet."""
        self.dataset.to_parquet(file_path, index=False)
        logging.info(f"Dataset salvato in formato Parquet: {file_path}")

    def clean_data(self):
        """Esegue una pulizia completa dei dati."""
        df = self.dataset
        
        # Rimuove duplicati
        df = DataCleaner.remove_duplicates(df)

        # Rimuove righe dove 'data_disdetta' non è nullo
        df = DataCleaner.remove_disdette(df)
        
        # Fix province e comuni
        df = DataFix.fill_province_comuni(df, self.path_province, self.path_comuni)

        # Rimuove colonne con valori mancanti sopra la soglia
        df = DataCleaner.remove_missing_values_rows(df, self.missing_threshold)
        
        # Aggiunge durata della visita e riempie le durate mancanti
        df = DataFix.add_durata_visita(df)
        df = DataFix.fill_durata_visita(df)
        
        # Aggiunge l'età del paziente
        df = DataFix.add_eta_paziente(df)

        # Aggiunge la fascia d'età
        df = DataFix.add_fascia_eta_column(df)

        # Identifica e gestisce gli outlier
        df = DataCleaner.update_dataset_with_outliers(df)

        # Rimuove colonne non più necessarie
        #columns = ['ora_inizio_erogazione', 'ora_fine_erogazione',
        #           'id_prenotazione', 'id_paziente', 'regione_residenza', 'codice_regione_residenza',
        #           'asl_residenza', 'codice_asl_residenza', 'provincia_residenza',
        #           'comune_residenza', 'codice_comune_residenza', 'descrizione_attivita',
        #           'data_contatto', 'regione_erogazione', 'asl_erogazione', 'provincia_erogazione',
        #           'tipologia_struttura_erogazione', 'id_professionista_sanitario',
        #           'tipologia_professionista_sanitario', 'codice_tipologia_professionista_sanitario']

        columns = ['ora_inizio_erogazione', 'ora_fine_erogazione',
                   'id_prenotazione', 'id_paziente', 'regione_residenza', 'codice_regione_residenza',
                   'asl_residenza', 'provincia_residenza',
                   'comune_residenza', 'codice_comune_residenza', 'descrizione_attivita',
                   'data_contatto', 'regione_erogazione', 'asl_erogazione', 'provincia_erogazione',
                   'id_professionista_sanitario',
                   'tipologia_professionista_sanitario']
        df = DataCleaner.remove_columns(df, columns)

        # Gestisce i valori mancanti nel dataset, riempiendo i valori mancanti in base alla media
        df = DataCleaner.handle_missing_values(df)

        # Imposta il dataset pulito
        self.set_dataset(df)
        logging.info("Pulizia completa del dataset eseguita con successo.")

        featureExtractor = FeatureExtractor(self.dataset)
        featureExtractor.run_analysis()
        df = featureExtractor.get_dataset()
        self.set_dataset(df)

        self.log_missing_values()

        featureSelection = FeatureSelection(self.dataset)
        featureSelection.execute_feature_selection(threshold=0.85,remove_others_colum_by_threshold=True)
        df = featureSelection.get_dataset()
        self.set_dataset(df)

        #TODO: dopo aver risolto la problematica che mancano 7000 righe con valore, rimuovere
        columns = ['incremento_percentuale',]
        df = DataCleaner.remove_columns(df, columns)
        ##

        self.set_dataset(df)
        self.log_missing_values()
        self.save_dataset()

        clustering = Clustering(self.dataset, n_clusters=4, algorithm='kmeans', scale_data=True, target_column='fascia_eta')
        clustering.run_full_clustering_analysis()

