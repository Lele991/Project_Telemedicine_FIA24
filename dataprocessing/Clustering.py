import logging
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples
from sklearn.decomposition import PCA
from collections import Counter

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Clustering:
    def __init__(self, df, n_clusters=4, algorithm='kmeans', columns=None, scale_data=True, target_column=None, max_iter=300):
        """
        Inizializza la classe con il dataset e i parametri di clustering.
        target_column: opzionale, usata per calcolare la purezza del clustering.
        max_iter: numero massimo di iterazioni per l'algoritmo KMeans.
        """
        logging.info(f"Inizializzazione della classe di clustering con {algorithm}.")
        self.df = df.copy()
        self.n_clusters = n_clusters
        self.algorithm = algorithm
        self.columns = columns if columns else df.columns.tolist()  # Usa tutte le colonne se non specificato
        self.scale_data = scale_data
        self.cluster_labels = None
        self.preprocessed_df = None
        self.target_column = target_column  # Colonna target per calcolo della purezza
        self.max_iter = max_iter
        self.original_fascia_eta = None  # Salva la colonna 'fascia_eta' originale se presente

    def preprocess_data(self, pca_components=None, remove_outliers=False):
        """
        Preprocessa i dati per l'analisi, gestisce le colonne categoriali con encoding e scala i dati numerici.
        Applica PCA opzionalmente e rimuove outlier se richiesto.
        """
        logging.info("Inizio preprocessing dei dati.")
        logging.info(f"Dimensione iniziale del dataset: {self.df.shape}")

        # Filtra per le colonne specificate, se esistono
        df_cluster = self.df[self.columns] if self.columns else self.df.copy()

        # Seleziona le colonne categoriali e numeriche
        categorical_columns = df_cluster.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_columns = df_cluster.select_dtypes(include=['int64', 'float64']).columns.tolist()

        logging.info(f"Colonne categoriali: {categorical_columns}")
        logging.info(f"Colonne numeriche: {numerical_columns}")

        # Salva una copia della colonna originale per la purezza (se presente)
        if 'fascia_eta' in df_cluster.columns:
            self.original_fascia_eta = df_cluster['fascia_eta'].copy()

        # Imputazione dei valori mancanti
        if categorical_columns:
            for col in categorical_columns:
                mode_value = df_cluster[col].mode()[0]
                df_cluster[col].fillna(mode_value, inplace=True)
        
        if numerical_columns:
            for col in numerical_columns:
                mean_value = df_cluster[col].mean()
                df_cluster[col].fillna(mean_value, inplace=True)

        # One-Hot Encoding delle colonne categoriali
        if categorical_columns:
            logging.info(f"Eseguo One-Hot Encoding delle colonne categoriali: {categorical_columns}")
            df_cluster = pd.get_dummies(df_cluster, columns=categorical_columns, drop_first=False)
        
        # Rimozione degli outlier se richiesto
        if remove_outliers:
            df_cluster = self.remove_outliers(df_cluster)

        # Scaling delle colonne numeriche
        if self.scale_data and numerical_columns:
            scaler = StandardScaler()
            df_cluster[numerical_columns] = scaler.fit_transform(df_cluster[numerical_columns])

        # Riduzione dimensionale con PCA
        if pca_components is not None:
            logging.info(f"Riduzione dimensionale tramite PCA a {pca_components} componenti")
            pca = PCA(n_components=pca_components)
            df_cluster = pd.DataFrame(pca.fit_transform(df_cluster), columns=[f'PC{i+1}' for i in range(pca_components)])
            logging.info(f"Dimensione dopo PCA: {df_cluster.shape}")

        self.preprocessed_df = df_cluster
        return df_cluster

    def remove_outliers(self, df_cluster):
        """
        Rimuove gli outlier dal dataset utilizzando l'IQR (Interquartile Range).
        """
        Q1 = df_cluster.quantile(0.25)
        Q3 = df_cluster.quantile(0.75)
        IQR = Q3 - Q1
        df_no_outliers = df_cluster[~((df_cluster < (Q1 - 1.5 * IQR)) | (df_cluster > (Q3 + 1.5 * IQR))).any(axis=1)]
        logging.info(f"Dimensione del dataset dopo la rimozione degli outlier: {df_no_outliers.shape}")
        return df_no_outliers

    def perform_clustering(self):
        """
        Esegue il clustering utilizzando l'algoritmo specificato e salva il modello.
        """
        # Esegui il preprocessing dei dati se non è già stato fatto
        if hasattr(self, 'preprocessed_df') and self.preprocessed_df is not None:
            logging.info("Preprocessing già eseguito, salto il preprocessing.")
            df_cluster = self.preprocessed_df
        else:
            logging.info("Preprocessing non ancora eseguito, avvio del preprocessing.")
            # Salva il dataframe preprocessato
            df_cluster = self.preprocess_data()

        if self.algorithm == 'kmeans':
            logging.info(f"Esecuzione di KMeans con {self.n_clusters} cluster.")
            model = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init=10, max_iter=self.max_iter, random_state=42)
            self.cluster_labels = model.fit_predict(df_cluster)

            # Salvataggio del modello di KMeans
            output_dir = 'saved_models'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_filepath = os.path.join(output_dir, 'kmeans_model.pkl')
            with open(model_filepath, 'wb') as file:
                pickle.dump(model, file)
            logging.info(f"Modello KMeans salvato con successo in {model_filepath}")

        self.df['cluster'] = self.cluster_labels
        logging.info(f"Clustering eseguito con successo. Cluster trovati: {np.unique(self.cluster_labels)}")

        return self.df

    def calculate_silhouette_per_cluster(self):
        """
        Calcola il Silhouette Score per ogni singolo cluster e il punteggio medio.
        """
        if self.cluster_labels is None:
            logging.warning("Clustering non ancora eseguito. Eseguire 'perform_clustering()' prima della valutazione.")
            return None

        # Esegui il preprocessing dei dati se non è già stato fatto
        if hasattr(self, 'preprocessed_df') and self.preprocessed_df is not None:
            logging.info("Preprocessing già eseguito, salto il preprocessing.")
            df_cluster = self.preprocessed_df
        else:
            logging.info("Preprocessing non ancora eseguito, avvio del preprocessing.")
            # Salva il dataframe preprocessato
            df_cluster = self.preprocess_data()

        silhouette_vals = silhouette_samples(df_cluster, self.cluster_labels)
        avg_silhouette = silhouette_vals.mean()

        # Calcola il silhouette medio per ciascun cluster
        silhouette_per_cluster = {}
        for cluster in np.unique(self.cluster_labels):
            cluster_silhouette_vals = silhouette_vals[self.cluster_labels == cluster]
            silhouette_per_cluster[cluster] = cluster_silhouette_vals.mean()
            logging.info(f"Silhouette Score per il cluster {cluster}: {silhouette_per_cluster[cluster]}")
        
        logging.info(f"Silhouette Score medio: {avg_silhouette}")
        return silhouette_per_cluster, avg_silhouette

    def calculate_purity(self):
        """
        Calcola la purezza di ogni cluster e la purezza media rispetto alla colonna 'fascia_eta' originale.
        """
        if not hasattr(self, 'original_fascia_eta'):
            logging.warning("La colonna 'fascia_eta' non è disponibile per calcolare la purezza.")
            return None

        # Crea un DataFrame temporaneo con cluster assegnati e la colonna target originale
        df_temp = pd.DataFrame({
            'cluster': self.cluster_labels,
            'fascia_eta': self.original_fascia_eta
        })

        purity_per_cluster = []
        total_correct = 0

        # Calcola la purezza per ogni cluster
        for cluster_id in np.unique(self.cluster_labels):
            cluster_data = df_temp[df_temp['cluster'] == cluster_id]
            most_common_class = cluster_data['fascia_eta'].mode()[0]
            correct = len(cluster_data[cluster_data['fascia_eta'] == most_common_class])
            total_correct += correct
            cluster_purity = correct / len(cluster_data)
            purity_per_cluster.append(cluster_purity)

        # Calcola la purezza media
        avg_purity = total_correct / len(df_temp)

        logging.info(f"Purezza media: {avg_purity}")
        return purity_per_cluster, avg_purity

    def calculate_purity_OLD(self):
        """
        Calcola la purezza di ogni singolo cluster e la purezza media.
        """
        if self.target_column is None:
            logging.warning("Colonna target non fornita, impossibile calcolare la purezza.")
            return None

        if self.cluster_labels is None:
            logging.warning("Clustering non ancora eseguito. Eseguire 'perform_clustering()' prima della valutazione.")
            return None
        
        # Verifica che la colonna target esista
        if self.target_column not in self.df.columns:
            logging.error(f"La colonna target '{self.target_column}' non esiste nel dataset.")
            return None

        purity_per_cluster = {}
        for cluster in np.unique(self.cluster_labels):
            cluster_points = self.df[self.cluster_labels == cluster]
            target_counts = Counter(cluster_points[self.target_column])
            max_class_count = max(target_counts.values())
            purity = max_class_count / len(cluster_points)
            purity_per_cluster[cluster] = purity
            logging.info(f"Purezza per il cluster {cluster}: {purity:.4f}")

        # Calcola la purezza media
        avg_purity = np.mean(list(purity_per_cluster.values()))
        logging.info(f"Purezza media: {avg_purity:.4f}")
        return purity_per_cluster, avg_purity

    def optimize_k(self, min_k=2, max_k=10):
        """
        Ottimizza il numero di cluster basandosi sul Silhouette Score.
        """

        # Esegui il preprocessing dei dati se non è già stato fatto
        if hasattr(self, 'preprocessed_df') and self.preprocessed_df is not None:
            logging.info("Preprocessing già eseguito, salto il preprocessing.")
            df_cluster = self.preprocessed_df
        else:
            logging.info("Preprocessing non ancora eseguito, avvio del preprocessing.")
            # Salva il dataframe preprocessato
            df_cluster = self.preprocess_data()

        best_k = min_k
        best_score = -1
        for k in range(min_k, max_k + 1):
            model = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=self.max_iter, random_state=42)
            cluster_labels = model.fit_predict(df_cluster)
            sil_score = silhouette_score(df_cluster, cluster_labels)
            logging.info(f"Silhouette Score per k={k}: {sil_score}")
            if sil_score > best_score:
                best_k = k
                best_score = sil_score

        logging.info(f"Il miglior numero di cluster è {best_k} con un Silhouette Score di {best_score}")
        self.n_clusters = best_k  # Aggiorna il numero di cluster ottimale
        return best_k

    def evaluate_clustering(self):
        """
        Valuta il clustering usando il Silhouette Score e Davies-Bouldin Index.
        Calcola anche il Silhouette per ogni singolo cluster e la purezza.
        """
        if self.cluster_labels is None:
            logging.warning("Clustering non ancora eseguito. Eseguire 'perform_clustering()' prima della valutazione.")
            return None

        # Esegui il preprocessing dei dati se non è già stato fatto
        if hasattr(self, 'preprocessed_df') and self.preprocessed_df is not None:
            logging.info("Preprocessing già eseguito, salto il preprocessing.")
            df_cluster = self.preprocessed_df
        else:
            logging.info("Preprocessing non ancora eseguito, avvio del preprocessing.")
            # Salva il dataframe preprocessato
            df_cluster = self.preprocess_data()

        # Calcolo del Silhouette Score complessivo e Davies-Bouldin Index
        sil_score = silhouette_score(df_cluster, self.cluster_labels)
        davies_bouldin = davies_bouldin_score(df_cluster, self.cluster_labels)
        
        # Calcola il Silhouette per ogni cluster
        silhouette_per_cluster, avg_silhouette = self.calculate_silhouette_per_cluster()

        # Calcola la purezza per ogni cluster
        purity_per_cluster, avg_purity = self.calculate_purity()

        logging.info(f"Silhouette Score medio: {avg_silhouette}")
        logging.info(f"Davies-Bouldin Index: {davies_bouldin}")
        logging.info(f"Purezza media: {avg_purity}")

        return {
            "Silhouette Score": sil_score,
            "Davies-Bouldin Index": davies_bouldin,
            "Silhouette per Cluster": silhouette_per_cluster,
            "Purezza per Cluster": purity_per_cluster,
            "Silhouette medio": avg_silhouette,
            "Purezza media": avg_purity
        }

    def plot_clusters(self):
        """
        Riduce dimensionalmente i dati con PCA a 2 dimensioni e genera un grafico dei cluster.
        """
        if self.cluster_labels is None:
            logging.warning("Clustering non ancora eseguito. Eseguire 'perform_clustering()' prima della visualizzazione.")
            return

        # Esegui il preprocessing dei dati se non è già stato fatto
        if hasattr(self, 'preprocessed_df') and self.preprocessed_df is not None:
            logging.info("Preprocessing già eseguito, salto il preprocessing.")
            df_cluster = self.preprocessed_df
        else:
            logging.info("Preprocessing non ancora eseguito, avvio del preprocessing.")
            # Salva il dataframe preprocessato
            df_cluster = self.preprocess_data()

        # Applica PCA per ridurre a 2 dimensioni
        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(df_cluster)

        # Crea un DataFrame con le componenti PCA
        df_pca = pd.DataFrame(data=pca_components, columns=['PCA1', 'PCA2'])
        df_pca['cluster'] = self.cluster_labels

        # Visualizza i cluster
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='PCA1', y='PCA2', hue='cluster', data=df_pca, palette="viridis", s=100, alpha=0.7)
        plt.title(f"Clustering usando PCA ({self.algorithm})")
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.legend(loc='best')
        plt.grid(True)

        # Salva il plot
        output_dir = 'graphs'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filepath = os.path.join(output_dir, 'cluster_plot.png')
        plt.savefig(filepath)
        plt.close()
        logging.info(f"Plot dei cluster salvato con successo in {filepath}")

    def run_full_clustering_analysis(self):
        """
        Esegue l'intera pipeline di clustering, inclusi il preprocessing, il clustering,
        la valutazione e la visualizzazione dei risultati.
        """
        logging.info(f"Dimensione iniziale del dataset: {self.df.shape}")
        logging.info("Inizio del processo di clustering completo.")

        # Esegui il preprocessing dei dati
        self.preprocess_data()

        # Ottimizza il numero di cluster se necessario
        self.optimize_k(min_k=2, max_k=10)

        # Esegue il clustering con il numero ottimale di cluster
        self.perform_clustering()

        # Valuta il clustering
        self.evaluate_clustering()

        # Visualizza i cluster con PCA
        self.plot_clusters()

        logging.info(f"Dimensione finale del dataset: {self.df.shape}")
        logging.info("Processo di clustering completo.")

