import logging
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureSelection:
    def __init__(self, df, categorical_columns=None, exclude_classes=None):
        self.df = df.copy()  # Memorizza il DataFrame per l'uso negli altri metodi
        self.colum_to_drop = None  # Memorizza le colonne rimosse
        self.exclude_classes = exclude_classes if exclude_classes else []  # Colonne da escludere

        if categorical_columns is None:
            # Usa tutte le colonne categoriali disponibili nel DataFrame se non specificato
            self.categorical_columns = self.df.columns.tolist()
            logging.info(f"Colonne categoriali rilevate automaticamente: {self.categorical_columns}")
        else:
            # Filtra solo le colonne categoriali che sono effettivamente presenti nel DataFrame
            self.categorical_columns = [col for col in categorical_columns if col in self.df.columns]
            logging.info(f"Colonne categoriali specificate dall'utente: {self.categorical_columns}")

        # Rimuovi le colonne da escludere dalla lista delle colonne categoriali
        self.categorical_columns = [col for col in self.categorical_columns if col not in self.exclude_classes]
        if self.exclude_classes:
            logging.info(f"Colonne escluse dal calcolo della V di Cramèr: {self.exclude_classes}")

    def get_dataset(self):
        return self.df
    
    def get_colum_to_drop(self):
        return self.colum_to_drop

    def calculate_cramers_v(self, column1, column2):
        logging.debug(f"Calcolo di Cramér's V tra {column1} e {column2}")
        contingency_table = pd.crosstab(self.df[column1], self.df[column2])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        n = contingency_table.sum().sum()
        r, k = contingency_table.shape
        denominator = n * (min(r-1, k-1))
        if denominator == 0:
            cramers_v = 0
        else:
            cramers_v = np.sqrt(chi2 / denominator)
        return cramers_v

    def create_correlation_matrix(self):
        if not self.categorical_columns:
            logging.warning("Nessuna colonna categoriale disponibile per creare la matrice di correlazione.")
            return pd.DataFrame()

        logging.info("Creazione della matrice di correlazione utilizzando Cramér's V")
        matrix = np.zeros((len(self.categorical_columns), len(self.categorical_columns)))

        for i in range(len(self.categorical_columns)):
            for j in range(len(self.categorical_columns)):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    matrix[i, j] = self.calculate_cramers_v(self.categorical_columns[i], self.categorical_columns[j])

        logging.info("Matrice di correlazione creata con successo")
        return pd.DataFrame(matrix, index=self.categorical_columns, columns=self.categorical_columns)

    def remove_perfectly_correlated_features(self, corr_matrix, threshold=1.0):
        logging.info(f"Rimozione delle colonne con correlazione perfetta ({threshold})")
        to_drop = set()

        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if corr_matrix.iloc[i, j] == threshold:
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    
                    # Ignora le colonne escluse
                    if col1 in self.exclude_classes or col2 in self.exclude_classes:
                        continue
                    
                    col1_cardinality = self.df[col1].nunique()
                    col2_cardinality = self.df[col2].nunique()
                    col1_nulls = self.df[col1].isnull().sum()
                    col2_nulls = self.df[col2].isnull().sum()

                    if col1_cardinality < col2_cardinality:
                        to_drop.add(col2)
                    elif col1_cardinality > col2_cardinality:
                        to_drop.add(col1)
                    else:
                        if col1_nulls > col2_nulls:
                            to_drop.add(col1)
                        else:
                            to_drop.add(col2)

        if to_drop:
            logging.info(f"Colonne rimosse per correlazione perfetta: {list(to_drop)}")
        else:
            logging.info("Nessuna colonna con correlazione perfetta trovata")

        self.categorical_columns = [col for col in self.categorical_columns if col not in to_drop]
        self.df.drop(columns=list(to_drop), inplace=True)
        self.colum_to_drop = list(to_drop)
        logging.info(f"Step 1: Colonne rimosse per correlazione perfetta: {self.colum_to_drop}")
        logging.info(f"Colonne rimanenti dopo la rimozione delle perfettamente correlate: {self.categorical_columns}")

    def remove_highly_correlated_features(self, corr_matrix, threshold=0.8):
        logging.info(f"Rimozione delle colonne con correlazione superiore alla soglia di {threshold}")
        to_drop = set()

        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if corr_matrix.iloc[i, j] > threshold:
                    col1 = corr_matrix.columns[i]
                    
                    # Ignora le colonne escluse
                    if col1 in self.exclude_classes:
                        continue

                    to_drop.add(col1)
        
        if to_drop:
            logging.info(f"Colonne rimosse per alta correlazione (soglia > {threshold}): {list(to_drop)}")
        else:
            logging.info("Nessuna colonna con alta correlazione trovata")

        self.df.drop(columns=list(to_drop), inplace=True)
        self.categorical_columns = [col for col in self.categorical_columns if col not in to_drop]
        self.colum_to_drop = self.colum_to_drop + (list(to_drop))
        print((list(to_drop)))
        logging.info(f"Step 2: Colonne rimosse per correlazione perfetta: {self.colum_to_drop}")
        logging.info(f"Colonne rimanenti dopo la rimozione delle altamente correlate: {self.categorical_columns}")

    def display_heatmap(self, corr_matrix, title, filename):
        if corr_matrix.empty:
            logging.warning("La matrice di correlazione è vuota. Nessuna heatmap sarà generata.")
            return

        logging.info(f"Creazione e salvataggio della heatmap: {title}")
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix.astype(float), annot=True, cmap="viridis", fmt=".2f", linewidths=0.5)
        plt.title(title)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()

        output_dir = 'graphs'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close()
        logging.info(f"Heatmap salvata con successo in {filepath}")

    def execute_feature_selection(self, threshold=0.8, remove_others_colum_by_threshold=False):
        logging.info("Inizio del processo di selezione delle caratteristiche")

        initial_corr_matrix = self.create_correlation_matrix()

        self.display_heatmap(initial_corr_matrix, "Matrice di Correlazione Originale", "heatmap_initial.png")
        
        self.remove_perfectly_correlated_features(initial_corr_matrix)
        
        if remove_others_colum_by_threshold:
            updated_corr_matrix = self.create_correlation_matrix()

            self.remove_highly_correlated_features(updated_corr_matrix, threshold)
        
        final_corr_matrix = self.create_correlation_matrix()

        self.display_heatmap(final_corr_matrix, "Matrice di Correlazione Finale", "heatmap_final.png")
        
        logging.info("Selezione delle caratteristiche completata con successo")
