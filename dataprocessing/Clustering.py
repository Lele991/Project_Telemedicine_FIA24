import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np
import logging
from collections import Counter
from kmodes.kmodes import KModes

class Clustering:
    def __init__(self, n_clusters=4):
        """
        Inizializza la classe per eseguire il clustering con KModes.
        :param n_clusters: Numero di cluster minimo per il KModes. Default = 4.
        """
        self.n_clusters = n_clusters
        self.kmodes = None
        self.labels = None
        self.silhouette_avg = None
        self.silhouette_values = None
        self.transformed_columns = []  # Tiene traccia delle colonne trasformate

    def check_null_values(self, dataset):
        """
        Verifica la presenza di righe con valori nulli nel dataset.
        :param dataset: Dataset da verificare.
        :return: None
        """
        null_rows = dataset.isnull().sum(axis=1)
        if null_rows.any():
            logging.warning(f"Ci sono {null_rows[null_rows > 0].count()} righe con valori nulli nel dataset.")
        else:
            logging.info("Nessun valore nullo trovato nel dataset.")

    def preprocess_data(self, dataset):
        """
        Trasforma le colonne categoriali in variabili numeriche usando Label Encoding.
        :param dataset: Dataset da preprocessare.
		:return: Dataset preprocessato (standardizzato) e i cluster.
        """
        logging.info("Inizio del preprocessamento delle colonne categoriali.")
        
        # Selezioniamo le colonne non numeriche
        #non_numeric_columns = dataset.select_dtypes(exclude=[np.number]).columns.tolist()
		# Rimuoviamo la colonna 'cluster' se presente
        clusters = dataset['cluster'].to_numpy()
        feature_columns = dataset.drop(columns=['cluster'])
        
        # Applica Label Encoding per le variabili categoriali
        for col in feature_columns.columns:
            if feature_columns[col].dtype == 'object':
                le = LabelEncoder()
                feature_columns[col] = le.fit_transform(feature_columns[col])
                self.transformed_columns.append((col, 'Label Encoding'))

		# Standardizzare solo le colonne numeriche (quelle già numeriche o codificate con Label Encoding)
        scaler = StandardScaler()
        X_standardized = scaler.fit_transform(feature_columns)
        X_standardized_df = pd.DataFrame(X_standardized)

        logging.info("Preprocessamento e standardizzazione completati.")
        return X_standardized_df, clusters

    def fit(self, dataset):
        """
        Esegue il clustering utilizzando KModes e calcola il silhouette score.
        :param dataset: Dataset preprocessato.
        :return: None
        """
        logging.info("Inizio del clustering KModes.")
        
        # Check se ci sono valori nulli nel dataset
        self.check_null_values(dataset)

        # Seleziona tutte le colonne per il clustering
        feature_columns = dataset.columns.tolist()

        # Esegui il clustering KModes
        self.kmodes = KModes(n_clusters=self.n_clusters, init='Huang', n_init=10, verbose=0)
        self.labels = self.kmodes.fit_predict(dataset[feature_columns])
        dataset['cluster'] = self.labels
        
        logging.info(f"Clustering labels: {self.labels}")
        logging.info(f"Clustering KModes completato con {self.n_clusters} cluster.")
        logging.info(f"Centroidi dei cluster: {self.kmodes.cluster_centroids_}")

        # Preprocessa il dataset per trasformare tutte le colonne categoriali per il calcolo del Silhouette Score
        X_standardized_df, clusters = self.preprocess_data(dataset)

        # Calcolo del Silhouette Score
        logging.info("Inizio calcolo del Silhouette Score.")
        #self.silhouette_avg = silhouette_score(dataset[feature_columns], self.labels)
        #self.silhouette_values = silhouette_samples(dataset[feature_columns], self.labels)
        self.silhouette_values = silhouette_samples(X_standardized_df, clusters)
        normalized_silhouette_vals = (self.silhouette_values - self.silhouette_values.min()) / (self.silhouette_values.max() - self.silhouette_values.min())
        self.silhouette_avg = np.mean(normalized_silhouette_vals)

        logging.info(f"Clustering completato con silhouette score medio: {self.silhouette_avg}.")

        X_standardized_df['cluster'] = self.labels
        return X_standardized_df
        
    def calculate_purity(self, dataset, label_column='incremento_classificato'):
        """
        Calcola la purezza del clustering basata sulla colonna specificata.
        Se la colonna è stata trasformata con One-Hot Encoding, aggrega i risultati.
        :param dataset: Dataset elaborato in ingresso dalla classe FeatureExtractor.
        :param label_column: Colonna di riferimento per calcolare la purezza. Default = 'incremento_classificato'.
        :return: Purezza del clustering.
        """
        logging.info("Calcolo della purezza del clustering.")

        # Verifica se la colonna originale esiste o è stata trasformata tramite One-Hot Encoding
        if label_column not in dataset.columns:
            logging.info(f"La colonna '{label_column}' non è presente direttamente. Verifico se è stata trasformata tramite One-Hot Encoding.")
            
            # Trova le colonne One-Hot Encoded corrispondenti a 'incremento_classificato'
            encoded_columns = [col for col in dataset.columns if label_column in col]
            
            if not encoded_columns:
                raise KeyError(f"La colonna '{label_column}' non è stata trovata né tra le colonne originali né tra quelle trasformate.")

            # Aggrega le colonne One-Hot Encoded: ciascuna riga prende l'etichetta della colonna con il valore massimo (1)
            dataset[label_column] = dataset[encoded_columns].idxmax(axis=1)

        # Trova il cluster con l'etichetta dominante
        cluster_labels = np.unique(self.labels)
        total_samples = dataset.shape[0]
        correct_predictions = 0

        for cluster in cluster_labels:
            cluster_indices = np.where(self.labels == cluster)[0]
            true_labels_in_cluster = dataset[label_column].iloc[cluster_indices]
            most_common_label = Counter(true_labels_in_cluster).most_common(1)[0][0]
            correct_predictions += sum(true_labels_in_cluster == most_common_label)

        purity = correct_predictions / total_samples
        logging.info(f"Purezza del clustering calcolata: {purity}")
        return purity

    def plot_clusters(self, dataset, cluster_column='cluster'):
        """
        Esegue il plot dei cluster generati e salva i plot nella cartella 'graphs'.
        Include il PCA plot e il Silhouette plot.
        :param dataset: Dataset preprocessato contenente le etichette del cluster.
        :param cluster_column: Nome della colonna per le etichette del cluster. Default = 'cluster'.
        :return: None
        """
        logging.info("Inizio della creazione dei grafici dei cluster.")

        # Creare la directory 'graphs' se non esiste
        output_dir = 'graphs'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Creata cartella '{output_dir}' per salvare i grafici.")

        # PCA per ridurre le dimensioni e visualizzare i cluster in 2D
        numeric_columns = dataset.drop(columns=[cluster_column]).select_dtypes(include=[np.number]).columns
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(dataset[numeric_columns])

        # Creare il plot PCA
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=dataset[cluster_column], cmap='Set1', edgecolor='k')
        
        # Aggiungere le etichette degli assi
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA su Dataset')
        
        # Aggiungere la legenda
        legend1 = plt.legend(*scatter.legend_elements(), title=cluster_column)
        plt.gca().add_artist(legend1)
        
        # Salvare il PCA plot
        pca_plot_path = os.path.join(output_dir, 'pca_clusters.png')
        plt.savefig(pca_plot_path)
        plt.close()
        logging.info(f"PCA plot salvato con successo in '{pca_plot_path}'.")

        # Plot del silhouette score
        plt.figure(figsize=(10, 6))
        y_lower = 10
        for i in range(self.n_clusters):
            ith_cluster_silhouette_values = self.silhouette_values[self.labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            plt.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values)
            plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10  # Spazio tra i grafici

        plt.title(f"Silhouette plot per i {self.n_clusters} cluster")
        plt.xlabel("Valore del Silhouette coefficient")
        plt.ylabel("Cluster")
        plt.axvline(x=self.silhouette_avg, color="red", linestyle="--")
        silhouette_plot_path = os.path.join(output_dir, 'silhouette_plot.png')
        plt.savefig(silhouette_plot_path)
        plt.close()
        logging.info(f"Silhouette plot salvato con successo in '{silhouette_plot_path}'.")


    def run_clustering(self, dataset, label_column='incremento_classificato', excluded_columns=None):
        """
        Esegue l'intero processo di clustering: clustering, silhouette score, purity e plot.
        :param dataset: Dataset elaborato in ingresso dalla classe FeatureExtractor.
        :param label_column: Colonna di riferimento per la purezza del cluster.
        :param excluded_columns: Lista di colonne da escludere dal clustering.
        :return: None
        """
        # Rimuovi le colonne specificate da `excluded_columns` se esistono
        if excluded_columns:
            dataset = dataset.drop(columns=excluded_columns, errors='ignore')
            logging.info(f"Colonne escluse dal clustering: {excluded_columns}")

        # Esegui clustering utilizzando tutte le colonne trasformate
        dataset_clustered = self.fit(dataset)
        print(dataset_clustered)
        print(dataset_clustered.columns)
        print(dataset_clustered.info())

        # Calcola purezza
        purity = self.calculate_purity(dataset, label_column)

        # Plot dei cluster
        self.plot_clusters(dataset_clustered)

        logging.info(f"Clustering KModes completato. Numero di Cluster: {self.n_clusters}, Silhouette Score Medio: {self.silhouette_avg}, Purezza: {purity}")
