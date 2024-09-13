import os
from flask import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPlot:
    
    def __init__(self, df):
        """
        Inizializza la classe con il dataset.
        :param df: Il dataset pandas
        """
        self.df = df
        # Crea la directory graphs se non esiste
        if not os.path.exists('graphs'):
            os.makedirs('graphs')
    
    def save_plot(self, plt, filename):
        """
        Salva il grafico nella directory 'graphs'.
        :param plt: L'oggetto plt di matplotlib
        :param filename: Nome del file in cui salvare il grafico
        """
        filepath = os.path.join('graphs', filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
    
    def plot_cluster_distribution(self):
        """
        Plot della distribuzione dei cluster
        """
        plt.figure(figsize=(8, 6))
        sns.countplot(x='cluster', data=self.df, palette='Set2')
        plt.title('Distribuzione dei Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Conteggio')
        self.save_plot(plt, 'cluster_distribution.png')

    def plot_sex_distribution_by_cluster(self):
        """
        Plot della distribuzione del sesso nei cluster usando la colonna 'sesso'.
        """

        plt.figure(figsize=(10, 6))
        
        if 'sesso_female' in self.df.columns and 'sesso_male' in self.df.columns:
            df_sex = self.df.groupby('cluster')[['sesso_female', 'sesso_male']].sum()
            df_sex.plot(kind='bar', stacked=True)
        elif 'sesso' not in self.df.columns:
            sns.countplot(x='cluster', hue='sesso', data=self.df, palette='Set2')

        plt.title('Distribuzione di Sesso nei Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Conteggio')
        plt.legend(title='Sesso')
        self.save_plot(plt, 'sex_distribution_by_cluster.png')

    def plot_age_distribution_by_cluster(self):
        """
        Plot della distribuzione delle fasce d'età nei cluster
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='cluster', y='fascia_eta', data=self.df, palette='Set1')
        plt.title('Distribuzione delle Fasce d\'Età nei Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Fascia Età')
        self.save_plot(plt, 'age_distribution_by_cluster.png')

    def plot_visit_duration_by_cluster(self):
        """
        Plot della durata delle visite per cluster
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='cluster', y='durata_visita', data=self.df, palette='Set3')
        plt.title('Durata delle Visite per Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Durata Visita (minuti)')
        self.save_plot(plt, 'visit_duration_by_cluster.png')

    def plot_cluster_by_region(self):
        """
        Plot della distribuzione dei cluster per regione di residenza
        """
        plt.figure(figsize=(12, 6))
        sns.countplot(x='codice_regione_residenza', hue='cluster', data=self.df, palette='Set2')
        plt.title('Distribuzione dei Cluster per Regione di Residenza')
        plt.xlabel('Codice Regione Residenza')
        plt.ylabel('Conteggio')
        plt.xticks(rotation=90)
        self.save_plot(plt, 'cluster_by_region.png')

    def plot_cluster_by_structure_type(self):
        """
        Plot della distribuzione dei cluster per tipologia di struttura
        """
        plt.figure(figsize=(12, 6))
        sns.countplot(x='codice_tipologia_struttura_erogazione', hue='cluster', data=self.df, palette='Set1')
        plt.title('Distribuzione dei Cluster per Tipologia di Struttura')
        plt.xlabel('Codice Tipologia Struttura')
        plt.ylabel('Conteggio')
        plt.xticks(rotation=90)
        self.save_plot(plt, 'cluster_by_structure_type.png')

    def plot_increment_by_cluster(self):
        """
        Plot della variabile incremento_classificato per cluster
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='cluster', y='incremento_classificato', data=self.df, palette='coolwarm')
        plt.title('Incremento Classificato per Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Incremento Classificato')
        self.save_plot(plt, 'increment_by_cluster.png')

    def plot_cluster_by_quarter(self):
        """
        Plot della distribuzione dei cluster per trimestre
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(x='trimestre', hue='cluster', data=self.df, palette='Set2')
        plt.title('Distribuzione dei Cluster per Trimestre')
        plt.xlabel('Trimestre')
        plt.ylabel('Conteggio')
        self.save_plot(plt, 'cluster_by_quarter.png')

    def generate_plots(self):
        """
        Genera tutti i grafici disponibili
        """
        self.plot_cluster_distribution()
        self.plot_sex_distribution_by_cluster()
        self.plot_age_distribution_by_cluster()
        self.plot_visit_duration_by_cluster()
        self.plot_cluster_by_region()
        self.plot_cluster_by_structure_type()
        self.plot_increment_by_cluster()
        self.plot_cluster_by_quarter()
        logging.info("Tutti i grafici sono stati generati e salvati nella cartella 'graphs'.")

