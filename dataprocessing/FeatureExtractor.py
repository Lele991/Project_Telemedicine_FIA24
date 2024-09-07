import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureExtractor:
    def __init__(self, dataset):
        """
        Inizializza la classe con il dataset già pulito e normalizzato.
        """
        self.dataset = dataset.copy()
        logging.info("FeatureExtractor inizializzato.")

    def get_dataset(self):
        return self.dataset

    def preprocess_data(self):
        """
        Preprocessa i dati per l'analisi, convertendo le date e creando nuove colonne per anno e trimestre.
        """
        logging.info("Inizio del preprocessamento dei dati.")
        
        # Conversione della colonna 'data_erogazione' in datetime, specificando utc=True per gestire i fusi orari
        self.dataset['data_erogazione'] = pd.to_datetime(self.dataset['data_erogazione'], errors='coerce', utc=True)

        # Gestione di eventuali date non valide
        if self.dataset['data_erogazione'].isnull().any():
            logging.warning("Ci sono valori mancanti o non validi in 'data_erogazione'. Saranno ignorati nel calcolo.")
        
        # Rimuove il fuso orario prima di convertire a trimestre
        self.dataset['trimestre'] = self.dataset['data_erogazione'].dt.tz_localize(None).dt.to_period('Q')
        #self.dataset['trimestre'] = self.dataset['data_erogazione'].dt.year.astype(str) + 'T' + self.dataset['data_erogazione'].dt.quarter.astype(str)
        
        # Creazione delle colonne 'anno'
        self.dataset['anno'] = self.dataset['data_erogazione'].dt.year
        
        logging.info("Preprocessamento dei dati completato.")
        return self.dataset

    def calculate_percentage_increments(self):
        """
        Calcola gli incrementi percentuali del numero di servizi per trimestre e codice descrizione attività.
        """
        logging.info("Inizio del calcolo degli incrementi percentuali.")

        # Raggruppamento per anno, trimestre e codice_descrizione_attivita, conteggio del numero di servizi
        grouped = self.dataset.groupby(['anno', 'trimestre', 'codice_descrizione_attivita']).size().reset_index(name='numero_servizi')

        # Ordinamento per 'codice_descrizione_attivita', 'anno', 'trimestre'
        grouped = grouped.sort_values(by=['codice_descrizione_attivita', 'anno', 'trimestre'])

        # Calcolo della differenza (incremento) del numero di servizi
        grouped['incremento'] = grouped.groupby('codice_descrizione_attivita')['numero_servizi'].diff()

        # Calcolo dell'incremento percentuale
        grouped['incremento_percentuale'] = (
            (grouped['incremento'] / grouped['numero_servizi'].shift(1)) * 100
        )

        # Riempire i NaN con 0 senza usare inplace=True
        grouped['incremento_percentuale'] = grouped['incremento_percentuale'].fillna(0)

        # Riempire anche gli incrementi dove il servizio precedente era 0
        grouped.loc[grouped['numero_servizi'].shift(1) == 0, 'incremento_percentuale'] = 0

        logging.info("Incrementi percentuali calcolati e aggiunti al dataset.")
        return grouped


    def classify_increment(self, value):
        """
        Classifica gli incrementi percentuali in base a soglie predefinite.
        """
        if value <= 5 and value >= 0:
            return 'constant_increment'
        elif value <= 15:
            return 'low_increment'
        elif value <= 40:
            return 'medium_increment'
        elif value > 40:
            return 'high_increment'
        else:
            return 'decrement'

    def apply_classification(self, grouped):
        """
        Applica la classificazione degli incrementi percentuali e unisce i risultati al dataset originale.
        """
        logging.info("Inizio della classificazione degli incrementi percentuali.")

        # Applica la classificazione
        grouped['incremento_classificato'] = grouped['incremento_percentuale'].apply(self.classify_increment)

        # Unione dei risultati al dataset originale
        self.dataset = pd.merge(self.dataset, grouped[['anno', 'trimestre', 'codice_descrizione_attivita', 'incremento_classificato']],
                                on=['anno', 'trimestre', 'codice_descrizione_attivita'], how='left')

        logging.info("Classificazione completata e aggiunta al dataset.")
        return self.dataset

    def plot_graphs(self, grouped):
        """
        Crea e salva i grafici della distribuzione degli incrementi e delle categorie di incremento,
        e l'andamento trimestrale delle teleassistenze.
        """
        logging.info("Inizio della creazione dei grafici.")
        output_dir = 'graphs'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        incremento_data = grouped['incremento_percentuale']
        column = 'attività'

        # Creazione della figura combinata
        plt.figure(figsize=(18, 12))

        # Distribuzione degli incrementi percentuali
        plt.subplot(2, 2, 1)
        sns.histplot(incremento_data, bins=30, kde=True, color='skyblue')
        plt.title(f'Distribuzione degli Incrementi Percentuali ({column})')
        plt.xlabel('Variazione Percentuale Media dei Servizi (%)')
        plt.ylabel('Numero di Occorrenze')  # numero di volte che un determinato incremento percentuale medio si verifica nel dataset
        plt.grid(True)

        # Boxplot degli incrementi percentuali
        plt.subplot(2, 2, 2)
        sns.boxplot(x=incremento_data, color='salmon')
        plt.title(f'Boxplot degli Incrementi Percentuali ({column})')
        plt.xlabel('Variazione Percentuale dei Servizi (%)')
        plt.grid(True)

        # Grafico dell'andamento trimestrale delle teleassistenze
        plt.subplot(2, 1, 2)
        trend = self.dataset.groupby('trimestre').size()
        
        # Generazione delle etichette personalizzate per trimestre e anno
        labels = []
        for period in trend.index:
            year = period.year
            quarter = period.quarter
            if quarter == 1:
                labels.append(f"{year}")
            else:
                labels.append(f"T{quarter}")

        # Plot del trend con le etichette personalizzate
        plt.plot(trend.index.astype(str), trend.values, marker='o', linestyle='-')
        plt.xticks(ticks=range(len(trend.index)), labels=labels, rotation=45)
        plt.title('Andamento delle Teleassistenze per Trimestre')
        plt.xlabel('Periodo (Anno e Trimestre)')
        plt.ylabel('Numero di Teleassistenze')
        plt.grid(True)

        # Salvataggio della figura combinata
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'combined_plot.png'))
        plt.close()

        logging.info(f"Grafici salvati con successo in '{output_dir}'.")

    def run_analysis(self):
        """
        Esegue l'intera pipeline di analisi: preprocessa i dati, calcola gli incrementi percentuali,
        classifica gli incrementi e genera i grafici.
        """
        logging.info("Inizio dell'analisi completa.")

        # Preprocessing dei dati
        self.preprocess_data()

        # Calcolo degli incrementi percentuali
        grouped = self.calculate_percentage_increments()

        # Creazione dei grafici
        self.plot_graphs(grouped)

        # Classificazione degli incrementi
        self.apply_classification(grouped)

        print(self.dataset)

        logging.info("Analisi completata.")