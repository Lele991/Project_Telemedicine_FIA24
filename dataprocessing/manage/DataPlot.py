import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import logging

class DataPlot:
    plt.set_loglevel('WARNING')

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

    def ensure_correct_column_types(self):
        """
        Assicura che i cluster e altre colonne categoriali siano trattati come tali.
        """
        categorical_columns = ['cluster', 'sesso', 'fascia_eta', 'regione_residenza', 
                               'descrizione_attivita', 'tipologia_struttura_erogazione', 
                               'trimestre', 'anno', 'incremento_classificato']
        
        for col in categorical_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str)  # Convert to string to avoid parsing issues

        # Ensure columns that should be numeric are treated as numeric
        if 'durata_visita' in self.df.columns:
            self.df['durata_visita'] = pd.to_numeric(self.df['durata_visita'], errors='coerce')

    def plot_cluster_distribution(self):
        """
        Plot della distribuzione dei cluster.
        """
        plt.figure(figsize=(8, 6))
        sns.countplot(x='cluster', data=self.df)
        plt.title('Distribuzione dei Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Conteggio')
        self.save_plot(plt, 'cluster_distribution.png')

    def plot_sex_distribution_by_cluster(self):
        """
        Plot della distribuzione del sesso nei cluster.
        """
        if 'sesso' not in self.df.columns:
            logging.warning("La colonna 'sesso' non esiste nel dataset.")
            return

        plt.figure(figsize=(10, 6))
        sns.countplot(x='cluster', hue='sesso', data=self.df, palette='Set2')
        plt.title('Distribuzione di Sesso nei Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Conteggio')
        plt.legend(title='Sesso')
        self.save_plot(plt, 'sex_distribution_by_cluster.png')

    def plot_age_distribution_by_cluster(self):
        """
        Plot della distribuzione delle fasce d'età nei cluster.
        """
        if 'fascia_eta' not in self.df.columns:
            logging.warning("La colonna 'fascia_eta' non esiste nel dataset.")
            return

        plt.figure(figsize=(10, 6))
        sns.violinplot(x='cluster', y='fascia_eta', data=self.df, palette='Set1', hue='cluster')
        plt.title('Distribuzione delle Fasce d\'Età nei Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Fascia Età')
        self.save_plot(plt, 'age_distribution_by_cluster.png')

    def plot_age_by_region(self):
        """
        Mostra la distribuzione delle visite per fascia d'età in base alla regione.
        """
        if 'fascia_eta' not in self.df.columns or 'regione_residenza' not in self.df.columns:
            logging.warning("Le colonne 'fascia_eta' o 'regione_residenza' non esistono nel dataset.")
            return

        plt.figure(figsize=(12, 8))
        df_grouped = self.df.groupby(['regione_residenza', 'fascia_eta'], observed=True).size().unstack(fill_value=0)
        df_grouped.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='Set1')
        plt.title('Distribuzione delle Visite per Fascia d\'Età e Regione')
        plt.xlabel('Regione')
        plt.ylabel('Numero di Visite')
        plt.xticks(rotation=90)
        self.save_plot(plt, 'age_by_region.png')

    def plot_cluster_by_region(self):
        """
        Cambia la tipologia di grafico per visualizzare tutte le regioni per cluster.
        """
        if 'regione_residenza' not in self.df.columns or 'cluster' not in self.df.columns:
            logging.warning("Le colonne 'regione_residenza' o 'cluster' non esistono nel dataset.")
            return

        plt.figure(figsize=(12, 8))
        df_grouped = self.df.groupby(['regione_residenza', 'cluster'], observed=True).size().unstack(fill_value=0)
        df_grouped.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='Set2')
        plt.title('Distribuzione dei Cluster per Regione di Residenza')
        plt.xlabel('Regione')
        plt.ylabel('Numero di Visite')
        plt.xticks(rotation=90)
        self.save_plot(plt, 'cluster_by_region.png')

    def plot_cluster_by_structure_type(self):
        """
        Plot della distribuzione dei cluster per tipologia di struttura.
        """
        if 'tipologia_struttura_erogazione' not in self.df.columns:
            logging.warning("La colonna 'tipologia_struttura_erogazione' non esiste nel dataset.")
            return

        plt.figure(figsize=(12, 6))
        sns.countplot(x='tipologia_struttura_erogazione', hue='cluster', data=self.df, palette='Set1')
        plt.title('Distribuzione dei Cluster per Tipologia di Struttura')
        plt.xlabel('Tipologia Struttura')
        plt.ylabel('Conteggio')
        plt.xticks(rotation=90)
        self.save_plot(plt, 'cluster_by_structure.png')

    def plot_increment_by_cluster(self):
        """
        Plot della variabile incremento_classificato per cluster.
        """
        if 'incremento_classificato' not in self.df.columns:
            logging.warning("La colonna 'incremento_classificato' non esiste nel dataset.")
            return

        plt.figure(figsize=(10, 6))
        sns.violinplot(x='cluster', y='incremento_classificato', data=self.df, hue='cluster', palette='coolwarm')
        plt.title('Incremento Classificato per Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Incremento Classificato')
        self.save_plot(plt, 'increment_by_cluster.png')

    def plot_cluster_by_quarter(self):
        """
        Plot della distribuzione dei cluster per trimestre.
        """
        if 'trimestre' not in self.df.columns:
            logging.warning("La colonna 'trimestre' non esiste nel dataset.")
            return

        plt.figure(figsize=(12, 6))
        sns.countplot(x='trimestre', hue='cluster', data=self.df, palette='Set2')
        plt.title('Distribuzione dei Cluster per Trimestre')
        plt.xlabel('Trimestre')
        plt.ylabel('Conteggio')
        plt.xticks(rotation=45, ha='right')
        self.save_plot(plt, 'cluster_by_quarter.png')

    def plot_cluster_by_year(self):
        """
        Plot della distribuzione dei cluster per anno.
        """
        if 'anno' not in self.df.columns:
            logging.warning("La colonna 'anno' non esiste nel dataset.")
            return

        plt.figure(figsize=(10, 6))
        sns.countplot(x='anno', hue='cluster', data=self.df, palette='Set3')
        plt.title('Distribuzione dei Cluster per Anno')
        plt.xlabel('Anno')
        plt.ylabel('Conteggio')
        self.save_plot(plt, 'cluster_by_year.png')

    def plot_cluster_by_professional(self, n=10):
        """
        Plot della distribuzione dei cluster per i primi 'n' professionisti sanitari più presenti nel dataset.
        """
        if 'id_professionista_sanitario' not in self.df.columns:
            logging.warning("La colonna 'id_professionista_sanitario' non esiste nel dataset.")
            return

        # Conta il numero di occorrenze di ogni professionista e seleziona i top 'n' professionisti
        top_professionisti = self.df['id_professionista_sanitario'].value_counts().nlargest(n).index

        # Filtra il DataFrame per includere solo i professionisti più frequenti
        df_filtered = self.df[self.df['id_professionista_sanitario'].isin(top_professionisti)]

        plt.figure(figsize=(12, 6))
        sns.countplot(x='id_professionista_sanitario', hue='cluster', data=df_filtered, palette='Set1')
        plt.title(f'Distribuzione dei Cluster per i primi {n} Professionisti Sanitari più presenti')
        plt.xlabel('ID Professionista Sanitario')
        plt.ylabel('Conteggio per Cluster')
        plt.xticks(rotation=90)
        self.save_plot(plt, f'cluster_by_professional_top_{n}.png')

    def plot_visit_duration_by_cluster(self):
        """
        Mostra la durata delle visite per cluster.
        """
        if 'durata_visita' not in self.df.columns:
            logging.warning("La colonna 'durata_visita' non esiste nel dataset.")
            return

        g = sns.FacetGrid(self.df, col="cluster", col_wrap=4, height=4, aspect=1)
        g.map(sns.histplot, "durata_visita", kde=True, color="purple")
        g.set_titles("Cluster {col_name}")
        g.set_axis_labels("Durata Visita (minuti)", "Conteggio")
        g.fig.suptitle("Durata delle Visite per Cluster (Facet Grid)", y=1.02)
        g.tight_layout()
        g.savefig('graphs/visit_duration_facet_by_cluster.png')

    def plot_visit_duration_strip_by_cluster(self):
        """
        Mostra la durata delle visite per cluster.
        """
        if 'durata_visita' not in self.df.columns:
            logging.warning("La colonna 'durata_visita' non esiste nel dataset.")
            return

        plt.figure(figsize=(10, 6))
        sns.stripplot(x='cluster', y='durata_visita', data=self.df, jitter=0.25, palette='Set2', hue='cluster', size=2.5)
        plt.title('Durata delle Visite per Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Durata Visita (minuti)')
        self.save_plot(plt, 'visit_duration_strip_by_cluster.png')

    def plot_visit_duration_by_sex(self):
        """
        Mostra la durata delle visite per sesso.
        """
        if 'durata_visita' not in self.df.columns or 'sesso' not in self.df.columns:
            logging.warning("Le colonne 'durata_visita' o 'sesso' non esistono nel dataset.")
            return

        plt.figure(figsize=(8, 6))
        sns.stripplot(x='sesso', y='durata_visita', data=self.df, jitter=True, hue='sesso', palette='Set2', size=2.5)
        plt.title('Durata delle Visite per Sesso')
        plt.xlabel('Sesso')
        plt.ylabel('Durata Visita (minuti)')
        self.save_plot(plt, 'visit_duration_by_sex.png')

    def plot_visit_duration_by_region_and_sex(self):
        """
        Mostra la durata delle visite per regione e sesso.
        """
        if 'durata_visita' not in self.df.columns or 'sesso' not in self.df.columns or 'regione_residenza' not in self.df.columns:
            logging.warning("Le colonne necessarie non esistono nel dataset.")
            return

        plt.figure(figsize=(12, 8))
        sns.violinplot(x='regione_residenza', y='durata_visita', hue='sesso', data=self.df, split=True, inner=None, palette='Set2')
        sns.stripplot(x='regione_residenza', y='durata_visita', hue='sesso', data=self.df, jitter=True, dodge=True, palette='Set2', marker='o', size=3, alpha=0.5)
        plt.xticks(rotation=90)
        plt.title('Durata delle Visite per Regione e Sesso')
        plt.xlabel('Regione di Residenza')
        plt.ylabel('Durata Visita (minuti)')
        plt.legend(title='Sesso', bbox_to_anchor=(1, 1))
        self.save_plot(plt, 'visit_duration_by_region_and_sex.png')

    def plot_visits_by_age_and_sex(self):
        """
        Mostra il numero di visite per fascia d'età e sesso.
        """
        if 'fascia_eta' not in self.df.columns or 'sesso' not in self.df.columns:
            logging.warning("Le colonne 'fascia_eta' o 'sesso' non esistono nel dataset.")
            return

        plt.figure(figsize=(12, 6))
        sns.countplot(x='fascia_eta', hue='sesso', data=self.df, palette='Set2')
        plt.title('Numero di Visite per Fascia d\'Età e Sesso')
        plt.xlabel('Fascia d\'Età')
        plt.ylabel('Conteggio Visite')
        self.save_plot(plt, 'visits_by_age_and_sex.png')

    def plot_visits_by_year_and_region(self):
        """
        Mostra la distribuzione delle visite per anno e regione.
        """
        if 'anno' not in self.df.columns or 'regione_residenza' not in self.df.columns:
            logging.warning("Le colonne 'anno' o 'regione_residenza' non esistono nel dataset.")
            return

        plt.figure(figsize=(12, 8))
        sns.countplot(x='anno', hue='regione_residenza', data=self.df, palette='Set3')
        plt.title('Distribuzione delle Visite per Anno e Regione')
        plt.xlabel('Anno')
        plt.ylabel('Conteggio Visite')
        self.save_plot(plt, 'visits_by_year_and_region.png')

    def plot_visit_duration_by_age_and_cluster(self):
        """
        Mostra la durata delle visite per fascia d'età e cluster.
        """
        if 'fascia_eta' not in self.df.columns or 'durata_visita' not in self.df.columns or 'cluster' not in self.df.columns:
            logging.warning("Le colonne necessarie non esistono nel dataset.")
            return

        g = sns.FacetGrid(self.df, col='fascia_eta', hue='cluster', col_wrap=4, height=4)
        g.map(sns.histplot, 'durata_visita', kde=False, multiple="stack")
        g.add_legend(title='Cluster')
        g.set_axis_labels('Durata Visita (minuti)', 'Conteggio')
        g.fig.suptitle('Durata delle Visite per Fascia d\'Età e Cluster', y=1.02)
        g.tight_layout()
        self.save_plot(plt, 'visit_duration_by_age_and_cluster.png')

    def plot_total_professionals_per_region(self):
        """
        Visualizza il numero di professionisti sanitari unici per ogni regione.
        """
        if 'regione_residenza' not in self.df.columns or 'id_professionista_sanitario' not in self.df.columns:
            logging.warning("Le colonne necessarie non esistono nel dataset.")
            return

        # Conta il numero di professionisti sanitari unici per regione
        professional_counts = self.df.groupby('regione_residenza')['id_professionista_sanitario'].nunique().sort_values(ascending=False)

        # Crea il barplot
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=professional_counts.index, y=professional_counts.values, hue=professional_counts.index, palette='Set3')

        # Aggiungi etichette e titolo
        plt.title('Numero Totale di Professionisti Sanitari Unici per Regione')
        plt.xlabel('Regione di Residenza')
        plt.ylabel('Numero di Professionisti Sanitari Unici')
        plt.xticks(rotation=90)

        # Aggiungi annotazioni su ogni barra per visualizzare il numero esatto
        for index, value in enumerate(professional_counts.values):
            ax.text(index, value + 50, str(value), ha='center', va='bottom')

        self.save_plot(plt, 'total_professionals_per_region.png')

    def plot_professionals_by_type_and_region(self):
        """
        Visualizza il numero di professionisti sanitari per tipologia in ogni regione.
        """
        if 'regione_residenza' not in self.df.columns or 'tipologia_professionista_sanitario' not in self.df.columns:
            logging.warning("Le colonne necessarie non esistono nel dataset.")
            return

        # Raggruppa i dati per regione e tipologia di professionista sanitario, quindi conta le occorrenze
        professional_counts = self.df.groupby(['regione_residenza', 'tipologia_professionista_sanitario']).size().unstack(fill_value=0)

        # Crea il barplot impilato (stacked bar plot)
        plt.figure(figsize=(12, 8))
        professional_counts.plot(kind='bar', stacked=True, colormap='Set3', figsize=(12, 8))

        # Aggiungi etichette e titolo
        plt.title('Numero di Professionisti Sanitari per Tipologia e Regione')
        plt.xlabel('Regione di Residenza')
        plt.ylabel('Numero di Professionisti Sanitari')
        plt.xticks(rotation=90)

        self.save_plot(plt, 'professionals_by_type_and_region.png')


    def generate_plots(self):
        """
        Genera tutti i grafici disponibili, ottimizzati.
        """
        self.ensure_correct_column_types()

        # Inizio generazione plot
        self.plot_cluster_distribution()
        self.plot_sex_distribution_by_cluster()
        self.plot_age_distribution_by_cluster()
        self.plot_age_by_region()
        self.plot_cluster_by_region()
        self.plot_cluster_by_structure_type()
        self.plot_increment_by_cluster()
        self.plot_cluster_by_quarter()
        self.plot_cluster_by_year()
        self.plot_cluster_by_professional(n=10)
        self.plot_visit_duration_by_cluster()
        self.plot_visit_duration_strip_by_cluster()
        self.plot_visit_duration_by_sex()
        self.plot_visit_duration_by_region_and_sex()
        self.plot_visits_by_age_and_sex()
        self.plot_visits_by_year_and_region()
        self.plot_visit_duration_by_age_and_cluster()
        self.plot_total_professionals_per_region()
        self.plot_professionals_by_type_and_region()

        logging.info("Tutti i grafici sono stati generati e salvati nella cartella 'graphs'.")
