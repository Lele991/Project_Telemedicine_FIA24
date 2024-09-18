from datetime import datetime
import os
from flask import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.model_selection import train_test_split
import numpy as np
import logging
from collections import Counter
from kmodes.kmodes import KModes
from kneed import KneeLocator

class Clustering:
    def __init__(self, n_clusters=4, use_one_hot=False):
        """
        Inizializza la classe per eseguire il clustering con KModes.
        :param n_clusters: Numero di cluster minimo per il KModes. Default = 4.
        :param use_one_hot: Se True, usa One-Hot Encoding; altrimenti usa Label Encoding.
        """
        self.n_clusters = n_clusters
        self.kmodes = None
        self.labels = None
        self.silhouette_avg = None
        self.silhouette_values = None
        self.purity = None
        self.final_metric = None
        self.transformed_columns = []  # Tiene traccia delle colonne trasformate
        self.use_one_hot = use_one_hot  # Controlla se usare One-Hot o Label Encoding
        self.dataset_clustered = None
        self.dataset_with_cluster = None

    def get_dataset_clustered(self):
        """
        Restituisce il dataset con le etichette del cluster.
        :return: Dataset con le etichette del cluster.
        """
        return self.dataset_clustered
    
    def get_dataset_with_cluster(self):
        return self.dataset_with_cluster

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

    def stratified_downsample(self, dataset, target_column, fraction=0.3):
        """
        Riduce il dataset mantenendo la proporzione delle classi nel target.
        :param target_column: La colonna in base alla quale effettuare il campionamento stratificato.
        :param fraction: La frazione del dataset da mantenere (0.1 = 10%)
        :return: Un dataset ridotto
        """
        df_reduced, _ = train_test_split(dataset, test_size=1-fraction, stratify=dataset[target_column], random_state=42)
        return df_reduced

    def elbow_method(self, dataset, min_clusters=2, max_clusters=10, threshold=0.05):
        """
        Esegue l'Elbow Method per determinare il numero ottimale di cluster.
        :param dataset: Dataset preprocessato.
        :param min_clusters: Numero minimo di cluster.
        :param max_clusters: Numero massimo di cluster.
        :param threshold: Threshold per individuare la differenza significativa tra distorsioni.
        :return: Il numero ottimale di cluster (k).
        """
        logging.info("Inizio dell'Elbow Method per determinare il numero ottimale di cluster.")

        # Crea la directory 'graphs' se non esiste
        output_dir = 'graphs'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Creata cartella '{output_dir}' per salvare i grafici.")

        distortions = []
        dataset = self.stratified_downsample(dataset, 'incremento_classificato')

        # Calcola la distorsione per ogni valore di k
        for k in range(min_clusters, max_clusters + 1):
            kmodes = KModes(n_clusters=k, init='Huang', n_init=5, verbose=0)
            labels = kmodes.fit_predict(dataset)
            distortions.append(kmodes.cost_)
            logging.info(f"Distortion per k={k}: {kmodes.cost_}")

        # Generare il plot dell'Elbow Method
        plt.figure(figsize=(8, 6))
        plt.plot(range(min_clusters, max_clusters + 1), distortions, 'bo-')
        plt.xticks(ticks=range(min_clusters, max_clusters + 1))
        plt.xlabel('Numero di Cluster (k)')
        plt.ylabel('Distortion')
        plt.title('Elbow Method per determinare il numero ottimale di cluster')
        elbow_plot_path = os.path.join(output_dir, 'elbow_method.png')
        plt.savefig(elbow_plot_path)
        plt.close()

        logging.info(f"Elbow plot salvato con successo in '{elbow_plot_path}'.")

        # Calcola la differenza di distorsione tra k consecutivi
        distortion_diffs = np.diff(distortions)
        logging.info(f"Differenze di distorsione: {distortion_diffs}")

        # Trova il punto dove la differenza tra le distorsioni diminuisce di meno del threshold
        for i in range(len(distortion_diffs) - 1):
            if abs(distortion_diffs[i + 1] - distortion_diffs[i]) < threshold:
                optimal_k = i + min_clusters + 1
                break
        else:
            optimal_k = 4

        logging.info(f"Numero ottimale di cluster secondo il metodo Elbow: {optimal_k}")

        # Aggiorna il numero di cluster
        self.n_clusters = optimal_k

    def preprocess_data(self, dataset):
        """
        Trasforma le colonne categoriali in variabili numeriche usando Label Encoding o One-Hot Encoding.
        :param dataset: Dataset da preprocessare.
        :return: Dataset preprocessato (standardizzato) e i cluster.
        """
        logging.info("Inizio del preprocessamento delle colonne categoriali.")
        
        # Rimuoviamo la colonna 'cluster' dal dataset
        clusters = dataset['cluster'].to_numpy()
        new_columns = dataset.drop(columns=['cluster'])
        
        # Scegli se usare One-Hot Encoding o Label Encoding
        if self.use_one_hot:
            logging.info("Utilizzo di One-Hot Encoding per le variabili categoriali.")
            # Identifica le colonne categoriali da trasformare
            categorical_cols = new_columns.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Applica One-Hot Encoding alle colonne categoriali
            new_columns = pd.get_dummies(new_columns, columns=categorical_cols, drop_first=False)
            self.transformed_columns.append((categorical_cols, 'One-Hot Encoding'))
        else:
            logging.info("Utilizzo di Label Encoding per le variabili categoriali.")
            # Applica Label Encoding alle colonne categoriali
            for col in new_columns.columns:
                if new_columns[col].dtype == 'object':
                    le = LabelEncoder()
                    new_columns[col] = le.fit_transform(new_columns[col])
                    self.transformed_columns.append((col, 'Label Encoding'))

        # Standardizzare solo le colonne numeriche (quelle già numeriche o codificate)
        #numeric_columns = new_columns.select_dtypes(include=[np.number])

        # Applica la standardizzazione solo ai dati numerici reali, non ai nomi delle colonne
        scaler = StandardScaler()
        df_standardized = scaler.fit_transform(new_columns)

        # Converti l'output standardizzato in DataFrame, mantenendo i nomi delle colonne originali
        df_standardized = pd.DataFrame(df_standardized)

        logging.info("Preprocessamento e standardizzazione completati.")
        
        return df_standardized, clusters

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
        new_columns = dataset.columns.tolist()

        # Esegui il clustering KModes
        self.kmodes = KModes(n_clusters=self.n_clusters, init='Huang', n_init=10, verbose=0)
        self.labels = self.kmodes.fit_predict(dataset[new_columns])
        dataset['cluster'] = self.labels
        self.dataset_with_cluster = dataset
        
        logging.info(f"Clustering completato con {self.n_clusters} cluster.")

        # Preprocessa il dataset per trasformare tutte le colonne categoriali per il calcolo del Silhouette Score
        df_standardized, clusters = self.preprocess_data(dataset)

        # Calcolo del Silhouette Score
        logging.info("Inizio calcolo del Silhouette Score.")
        self.silhouette_values = silhouette_samples(df_standardized, clusters)
        normalized_silhouette = (self.silhouette_values - self.silhouette_values.min()) / (self.silhouette_values.max() - self.silhouette_values.min())
        self.silhouette_avg = np.mean(normalized_silhouette)

        logging.info(f"Clustering completato con silhouette score medio: {self.silhouette_avg}.")
        
        df_standardized['cluster'] = self.labels
        return df_standardized
    
    def calculate_final_metric(self):
        """
        Calcola la metrica finale come la media tra purezza e silhouette_mean meno 0.05 volte il numero di cluster.
        Utilizza np.mean per calcolare la media.
        :return: La metrica finale.
        """
        # Calcola la media di purezza e silhouette_avg
        average_score = np.mean([self.purity, self.silhouette_avg])
        
        # Applica la penalizzazione basata sul numero di cluster
        penalty = 0.05 * self.n_clusters
        
        # Calcola la metrica finale
        self.final_metric = average_score - penalty
        
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
        self.purity = purity

    def plot_clusters_3d(self, dataset, cluster_column='cluster'):
        """
        Esegue il plot dei cluster generati e salva i plot nella cartella 'graphs' in 3D.
        Include il PCA plot in 3D.
        :param dataset: Dataset preprocessato contenente le etichette del cluster.
        :param cluster_column: Nome della colonna per le etichette del cluster. Default = 'cluster'.
        :return: None
        """
        logging.info("Inizio della creazione del grafico 3D dei cluster.")

        # Creare la directory 'graphs' se non esiste
        output_dir = 'graphs'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Creata cartella '{output_dir}' per salvare i grafici.")

        # PCA per ridurre le dimensioni e visualizzare i cluster in 3D
        numeric_columns = dataset.drop(columns=[cluster_column]).select_dtypes(include=[np.number]).columns
        pca = PCA(n_components=3)  # 3 componenti per il grafico 3D
        pca_result = pca.fit_transform(dataset[numeric_columns])

        # Crea un DataFrame con le componenti PCA
        df_pca = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2', 'PCA3'])
        df_pca['cluster'] = self.labels

        # Crea il plot 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Colori per i cluster
        colors = ['red', 'black', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta']

        # Filtra e plottare i cluster in modo separato
        for i in range(self.n_clusters):
            filtered_data = df_pca[df_pca['cluster'] == i]
            ax.scatter(filtered_data['PCA1'], filtered_data['PCA2'], filtered_data['PCA3'], 
                    color=colors[i % len(colors)], label=f'Cluster {i}', s=50, alpha=0.7)
        
        # Aggiungere titoli e etichette agli assi
        ax.set_title(f"Visualizzazione Cluster in 3D usando PCA ({self.n_clusters} cluster)")
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        ax.set_zlabel('PCA 3')
        plt.legend(loc='best')

        # Salvare il PCA plot in 3D
        pca_plot_path = os.path.join(output_dir, 'pca_clusters_3d.png')
        plt.savefig(pca_plot_path)
        plt.close()
        logging.info(f"PCA plot 3D salvato con successo in '{pca_plot_path}'.")

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

        # Crea un DataFrame con le componenti PCA
        df_pca = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
        df_pca['cluster'] = self.labels

        # Visualizza i cluster
        plt.figure(figsize=(10, 8))

        # Colori per i cluster
        colors = ['red', 'black', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta']
        
        # Filtra e plottare i cluster in modo separato
        for i in range(self.n_clusters):
            filtered_data = df_pca[df_pca['cluster'] == i]
            plt.scatter(filtered_data['PCA1'], filtered_data['PCA2'], 
                        color=colors[i % len(colors)], label=f'Cluster {i}')

        plt.title(f"Clustering usando PCA ({self.n_clusters} cluster)")
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.legend(loc='best')
        plt.grid(True)
        
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
                            0, ith_cluster_silhouette_values, alpha=0.7)
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
    
    def save_results(self, excluded_columns, used_columns):
        # Save the results to a JSON file
        results = {
            "data_elaborazione": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "colonne_escluse" : excluded_columns,
            "colonne_utilizzate": used_columns,
            "numero_di_cluster": self.n_clusters,
            "silhouette_score_medio": self.silhouette_avg,
            "purezza": self.purity,
            "metrica_finale": self.final_metric
        }

        output_dir = 'results'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Creata cartella '{output_dir}' per salvare i risultati.")

        results_path = os.path.join(output_dir, 'clustering_results.json')
        with open(results_path, 'w') as json_file:
            json.dump(results, json_file, indent=4)

        logging.info(f"Risultati del clustering salvati con successo in '{results_path}'.")

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

        # Esegui l'Elbow Method per trovare il numero ottimale di cluster
        self.elbow_method(dataset, min_clusters=2, max_clusters=6)


        # Esegui clustering utilizzando tutte le colonne trasformate
        dataset_clustered = self.fit(dataset)
        self.dataset_clustered = dataset_clustered
        print(dataset_clustered)
        print(dataset_clustered.columns)
        print(dataset_clustered.info())

        # Calcola purezza
        self.calculate_purity(dataset, label_column)

        # Calcolo metrica finale con penalty di 0.5
        self.calculate_final_metric()

        # Plot dei cluster
        self.plot_clusters(dataset_clustered)
        self.plot_clusters_3d(dataset_clustered)

        # Salva
        self.save_results(excluded_columns, dataset.columns.tolist())

        # Log the clustering results
        logging.info(f"Clustering KModes completato. Numero di Cluster: {self.n_clusters}, Silhouette Score Medio: {self.silhouette_avg}, Purezza: {self.purity}")
