import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureExtractor:
    def __init__(self, dataset):
        """
        Inizializza la classe con il dataset già pulito e normalizzato.
        """
        self.dataset = dataset.copy()
        self.incremento_percentuale_medio = None

    def get_dataset(self):
        """
        Ritorna il dataset attuale.
        """
        return self.dataset

    def calculate_percentage_increments(self):
        """Calcola gli incrementi percentuali di una colonna specificata e li aggiunge al dataset."""
        # Definizione delle colonne per il raggruppamento
        cols_grouped = ['anno', 'trimestre', 'codice_descrizione_attivita']
        
        # Raggruppamento dei dati e conteggio dei servizi
        df_grouped = self.dataset.groupby(cols_grouped).size().reset_index(name='numero_servizi')
        
        # Rimozione della colonna 'anno' per calcolare l'incremento per ciascun gruppo
        df_cols_no_anno = cols_grouped.copy()
        df_cols_no_anno.remove('anno')
        
        # Calcolo dell'incremento del numero di servizi per ciascun gruppo
        df_grouped['incremento'] = df_grouped.groupby(df_cols_no_anno)['numero_servizi'].diff()
        
        # Calcolo dell'incremento percentuale
        df_grouped['incremento_percentuale'] = df_grouped['incremento'] / df_grouped.groupby(df_cols_no_anno)['numero_servizi'].shift(1) * 100
        #df_grouped['incremento_percentuale'].fillna(0, inplace=True)

        self.incremento_percentuale_medio = df_grouped.groupby(df_cols_no_anno)['incremento_percentuale'].mean().reset_index()
        
        # Unisce i risultati al dataset originale
        new_cols = ['incremento_percentuale', 'numero_servizi']
        self.dataset = pd.merge(self.dataset, df_grouped[cols_grouped + new_cols],
                                on=cols_grouped, how='left')

    def add_trimestre_column(self):
        """
        Aggiunge una colonna che indica il trimestre di erogazione del servizio,
        gestendo correttamente le differenze di fuso orario.
        """
        # Assicuriamoci che la colonna 'data_erogazione' sia in formato datetime e UTC
        self.dataset['data_erogazione'] = pd.to_datetime(self.dataset['data_erogazione'], utc=True)
        
        # Rimuoviamo il fuso orario per evitare l'avviso
        self.dataset['data_erogazione'] = self.dataset['data_erogazione'].dt.tz_localize(None)
        
        # Creiamo una nuova colonna con il trimestre di erogazione del servizio
        self.dataset['trimestre'] = self.dataset['data_erogazione'].dt.to_period('T')

    def preprocess_data(self):
        """
        Preprocessa i dati per l'analisi, convertendo le date e creando nuove colonne per anno e trimestre.
        """
        # Conversione della colonna 'data_erogazione' in datetime e gestione dei fusi orari
        self.dataset['data_erogazione'] = pd.to_datetime(self.dataset['data_erogazione'], utc=True, errors='coerce')
        
        if self.dataset['data_erogazione'].isnull().any():
            print("Warning: Ci sono valori mancanti in 'data_erogazione'.")
        
        # Creazione delle colonne 'anno' e 'trimestre'
        self.dataset['anno'] = self.dataset['data_erogazione'].dt.year
        self.dataset['trimestre'] = ((self.dataset['data_erogazione'].dt.month - 1) // 3) + 1

    def compute_numero_servizi(self):
        """
        Calcola il numero di servizi per anno, trimestre e codice_descrizione_attivita.
        """
        # Raggruppamento per anno, trimestre e codice_descrizione_attivita
        cols_grouped = ['anno', 'trimestre', 'codice_descrizione_attivita']
        df_grouped = self.dataset.groupby(cols_grouped).size().reset_index(name='numero_servizi')
        self.dataset = pd.merge(self.dataset, df_grouped, on=cols_grouped, how='left')
        return df_grouped

    def plot_graph(self):
        """
        Crea e salva i grafici della distribuzione degli incrementi e delle categorie di incremento,
        e l'andamento trimestrale delle teleassistenze.
        """
        output_dir = 'graphs'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
        incremento_data = self.incremento_percentuale_medio['incremento_percentuale']
        column = 'attività'

        # Creazione della figura combinata
        plt.figure(figsize=(18, 12))

        # Distribuzione degli incrementi percentuali
        plt.subplot(2, 2, 1)
        sns.histplot(incremento_data, bins=30, kde=True, color='skyblue')
        plt.title(f'Distribuzione degli Incrementi Percentuali ({column})')
        plt.xlabel('Variazione Percentuale Media dei Servizi (%)')
        plt.ylabel('Numero di Occorrenze') # indica il numero di volte che un determinato incremento percentuale medio si verifica nel dataset
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
        # Generazione delle etichette personalizzate
        labels = []
        previous_year = None
        for period in trend.index:
            year = period.year
            quarter = period.quarter
            if quarter == 1:
                labels.append(f"{year}")
                previous_year = year
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

    def run_analysis(self):
        """
        Esegue l'intera pipeline di analisi: preprocessa i dati, calcola gli incrementi percentuali,
        aggiunge la colonna del trimestre, e genera i grafici richiesti.
        """
        # Preprocessing dei dati
        self.preprocess_data()
        
        # Calcolo degli incrementi percentuali
        self.calculate_percentage_increments()
        
        # Aggiunta della colonna del trimestre
        self.add_trimestre_column()
        
        # Generazione e salvataggio dei grafici
        self.plot_graph()
