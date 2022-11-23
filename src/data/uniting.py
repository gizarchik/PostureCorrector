import pandas as pd
from datetime import date

import src.config as CFG

def unite_datasets(new_data_path, new_person_path,
                   data_path=CFG.RAW_DATA_PATH, person_path=CFG.RAW_PERSON_PATH):
    '''
    Добавляет новые данные в старые.
    Копия старых данных сохраняется.
    '''
    new_data = pd.read_csv(new_data_path, index_col=0)
    new_person = pd.read_csv(new_person_path, index_col='_id')
    
    data = pd.read_csv(data_path, index_col=0)
    person = pd.read_csv(person_path, index_col='_id')
    
    today = date.today().strftime("%m_%d_%y")
    data.to_csv(CFG.ARCHIVE_PATH + 'data_'+today+'.csv')
    person.to_csv(CFG.ARCHIVE_PATH + 'person_'+today+'.csv')
    
    last_index = data.iloc[-1].name
    last_id = person.iloc[-1].name
    
    new_data.index += last_index + 1
    new_data['_id'] += last_id + 1
    new_person.index += last_id + 1
    
    data = pd.concat([data, new_data])
    person = pd.concat([person, new_person])
    
    data.to_csv(data_path)
    person.to_csv(person_path)
    