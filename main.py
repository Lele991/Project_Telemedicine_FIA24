import pandas as pd
from dataprocessing.ManageData import ManageData

#Dichiarazione del metodo per caricare il file dei dati
def load_data(file_path):
    df = pd.read_parquet(file_path)
    return df

#Specifica del percorso per raggiungere il dataset
def main():
    
    file_path_data = 'data/challenge_campus_biomedico_2024.parquet'
    file_path_province = 'data/italia/province.json'
    file_path_comuni = 'data/italia/comuni.json'
    
    
    #Imposto la soglia mancante per la futura eliminazione dei dati che la superano
    missing_threshold = 0.6

    # Carico il file dei dati nel Dataframe
    df = load_data(file_path_data)

    # Visualizzazione sul terminale del Dataframe
    # print(df)

    data = ManageData(df, file_path_province, file_path_comuni, missing_threshold)

    #Sostituisco i valori None(nulli) con "NaN" nel Dataframe
    data.replace_none_with_nan()

    #Eseguo una pulizia dei dati
    data.clean_data()
    
    #Definisco il metodo per recuperare il dataset dal file dei dati
    df = data.get_dataset()
    #print(df)
    
    #Visualizzo le colonne presenti nel Dataframe finale
    colonne = df.columns
    print("Colonne presenti nel DataFrame finale:")
    print(colonne.tolist())



if __name__ == "__main__":
    main()