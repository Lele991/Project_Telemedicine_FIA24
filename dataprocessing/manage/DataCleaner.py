import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.cluster import DBSCAN

class DataCleaner:
    def __init__(self, data):
        self.data = data

    def get_data(self):
        """Restituisce i dati elaborati."""
        return self.data

    def remove_duplicates(self):
        """Rimuovi i duplicati nei dati."""
        self.data.drop_duplicates(inplace=True)
        return self.dataset

    def remove_unnecessary_columns(self, null_threshold=0.6):
        """
        Rimuovi le colonne che hanno una percentuale di valori nulli superiore alla soglia specificata.
        
        Parametri:
        null_threshold (float): La soglia di nullità (da 0.0 a 1.0) oltre la quale una colonna viene rimossa.
                                Il valore predefinito è 0.6 (60%).
        """
        self.data = self.data.loc[:, self.data.isnull().mean() < null_threshold]

    def process_outliers_dbscan(self, eps=0.5, min_samples=5):
        """Rimuovi outlier utilizzando DBSCAN sui dati numerici."""
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.data['cluster'] = dbscan.fit_predict(self.data.select_dtypes(include=[float, int]))
        self.data = self.data[self.data['cluster'] != -1]
        self.data.drop(columns=['cluster'], inplace=True)

    def clean_full_data(self, missing_threshold=0.6):
        """Esegue una pulizia completa dei dati."""
        # Rimuove colonne con troppi valori mancanti
        self.remove_unnecessary_columns(null_threshold=missing_threshold)

        # Rimuove duplicati
        self.remove_duplicates()

        # Gestione degli outlier
        self.process_outliers_dbscan()

        return self.data
