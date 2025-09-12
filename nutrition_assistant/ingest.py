import os
from pathlib import Path
import pandas as pd
from . import minsearch

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "data.csv"
DATA_PATH = os.getenv("DATA_PATH", str(DEFAULT_DATA_PATH))

def load_index(data_path=DATA_PATH):
    df = pd.read_csv(data_path, sep=';', encoding='utf-8')  # если ваш CSV с ';'
    df.columns = df.columns.str.strip().str.lower()
    df = df.fillna("")
    documents = df.to_dict(orient="records")

    index = minsearch.Index(
        text_fields=[
            'food','serving_size_g','calories_kcal','protein_g','fat_g',
            'carbohydrates_g','vitamin_a_mg','vitamin_b6_mg','vitamin_b12_mg',
            'vitamin_c_mg','vitamin_d_mg','vitamin_e_mg','calcium_mg','iron_mg',
            'potassium_mg','magnesium_mg','selenium_mg','zinc_mg','iodine_mg',
            'allergens'
        ],
        keyword_fields=['id']
    )
    index.fit(documents)
    return index
