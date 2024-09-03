import pandas as pd
from dataprocessing.ManageData import ManageData

def load_data(file_path):
    # Load the data
    df = pd.read_parquet(file_path)
    return df


def main():
    # Specify the path to the Parquet file
    file_path_data = 'data/challenge_campus_biomedico_2024.parquet'
    file_path_province = 'data/italia/province.json'
    file_path_comuni = 'data/italia/comuni.json'
    
    # Set the missing threshold
    missing_threshold = 0.6

    # Load the Parquet file into a DataFrame
    df = load_data(file_path_data)

    # Print the DataFrame
    #print(df)

    data = ManageData(df, file_path_province, file_path_comuni, missing_threshold)
    data.replace_none_with_nan()
    data.clean_data()
    
    df = data.get_dataset()
    #print(df)

    colonne = df.columns
    print("Colonne presenti nel DataFrame finale:")
    print(colonne.tolist())



if __name__ == "__main__":
    main()