import sys
sys.path.append("../../../")
sys.path.append("../../../outboxml")

from outboxml import config
from outboxml.automl_manager import AutoMLConfig
from sqlalchemy import create_engine, text
import pandas as pd
import json

with open('configs/automl-titanic.json', encoding='utf-8') as f:
    auto_ml_config = AutoMLConfig.model_validate(json.load(f))

engine = create_engine(config.connection_params)
trigger = auto_ml_config.trigger
table_name = trigger['table_name']

query = f"""SELECT * FROM "{table_name}";"""
df = pd.read_sql(query, engine)

max_id = df['PASSENGERID'].max()
row_to_add = df.tail(1).replace({'PASSENGERID': max_id}, max_id+10)
row_to_add.to_sql(table_name, con=config.connection_params, if_exists='append', index=False)
print(row_to_add)
print('Строка успешно добавлена')