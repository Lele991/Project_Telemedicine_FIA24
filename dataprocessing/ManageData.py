import numpy as np
import pandas as pd
import logging

from dataprocessing.FeatureSelection import FeatureSelection
from dataprocessing.FeatureExtractor import FeatureExtractor
from dataprocessing.manage import DataCleaner, DataFix

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

        # Gestisce i valori mancanti nel dataset, riempiendo i valori mancanti in base alla media
        #df = DataCleaner.handle_missing_values(df)

        # Identifica e gestisce gli outlier
        df = DataCleaner.update_dataset_with_outliers(df)

        # Rimuove colonne non più necessarie
        columns = ['ora_inizio_erogazione', 'ora_fine_erogazione',
                   'id_prenotazione', 'id_paziente', 'regione_residenza', 'codice_regione_residenza',
                   'asl_residenza', 'codice_asl_residenza', 'provincia_residenza',
                   'comune_residenza', 'codice_comune_residenza', 'descrizione_attivita',
                   'data_contatto', 'regione_erogazione', 'asl_erogazione', 'provincia_erogazione',
                   'tipologia_struttura_erogazione', 'id_professionista_sanitario',
                   'tipologia_professionista_sanitario', 'codice_tipologia_professionista_sanitario']

        df = DataCleaner.remove_columns(df, columns)

        # Imposta il dataset pulito
        self.set_dataset(df)
        logging.info("Pulizia completa del dataset eseguita con successo.")

        featureExtractor = FeatureExtractor(self.dataset)
        featureExtractor.run_analysis()
        df = featureExtractor.get_dataset()
        self.set_dataset(df)

        featureSelection = FeatureSelection(df)
        featureSelection.execute_feature_selection()

