# classe principale per la gestione dei dati
# -->
# chaimata al file DataFix per popolare i campi mancanti
# chiamata al file DataCleaner per la pulizia dei dati


import numpy as np
import pandas as pd

from dataprocessing.manage import DataCleaner, DataFix

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

    def clean_data(self):
        """Esegue una pulizia completa dei dati."""
        df = self.dataset
        #columns = [
        # data inpiut 1/2:
        #    'id_prenotazione', 'id_paziente', 'codice_regione_residenza', 'codice_asl_residenza', 'codice_provincia_residenza',
        #    'codice_comune_residenza', 'descrizione_attivita', 'data_contatto', 'data_disdetta',
        # data inpiut 2/2:
        #    'codice_regione_erogazione', 'struttura_erogazione', 'tipologia_struttura_erogazione', 'id_professionista_sanitario',
        #    'tipologia_professionista_sanitario', 'codice_tipologia_professionista_sanitario', 'ora_inizio_erogazione', 'ora_fine_erogazione', 'data_erogazione'
        #]

        # Rimuove duplicati
        df = DataCleaner.remove_duplicates(df)

        # Rimuove righe dove 'data_disdetta' non è nullo
        df = DataCleaner.remove_disdette(df)
        
        # Fix province e comuni
        df = DataFix.fill_province_comuni(df, self.path_province, self.path_comuni)

        df = DataCleaner.remove_missing_values_rows(df, self.missing_threshold)
        
        df = DataFix.add_durata_visita(df)
        df = DataFix.fill_durata_visita(df)
        df = DataFix.add_eta_paziente(df)

        df = DataCleaner.update_dataset_with_outliers(df)

        columns = ['ora_inizio_erogazione', 'ora_fine_erogazione']
        df = DataCleaner.remove_columns(df, columns)

        self.set_dataset(df)