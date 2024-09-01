# classe principale per la gestione dei dati
# -->
# chaimata al file DataFix per popolare i campi mancanti
# chiamata al file DataCleaner per la pulizia dei dati


from dataprocessing.manage import DataFix

class ManageData:
    def __init__(self, dataset, path_province, path_comuni):
        self.dataset = dataset
        self.path_province = path_province
        self.path_comuni = path_comuni

    def fix_province(self):
        # Call the fetch_province_code_data and fetch_comuni_code_data function from DataFix.py
        # to get the dictionaries codice_to_provincia and provincia_to_codice
        codice_to_provincia, provincia_to_codice = DataFix.fetch_province_code_data(self.path_province)
        codice_to_comune, comune_to_codice = DataFix.fetch_comuni_code_data(self.path_comuni)

        print(codice_to_comune.get('1168'))
        print(provincia_to_codice.get('Napoli'))

        # Check if the dictionaries are not None
        if codice_to_provincia is not None and provincia_to_codice is not None and codice_to_comune is not None and comune_to_codice is not None:
            # Call the fill_province_comuni function from DataFix.py
            self.dataset = DataFix.fill_province_comuni(self.dataset, codice_to_provincia, provincia_to_codice, codice_to_comune, comune_to_codice)