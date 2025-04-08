import pandas as pd
import re

units = {
    "solid": ["gm", "g", "kg", "gram"],
    "liquid": ["ml", "l", "litre"],
    "unit": ["no", "number", "unit"]

}

def get_key_by_value(d, value):
    for key, values in d.items():
        if value.lower() in values:
            return key
    return None


data_items_df = pd.read_csv(r"D:\Rohit\ORG_SKUR\new_data\data-new-items-202410.csv")
master_df = pd.read_csv(r"/dump/master_filter_3_dynamic.csv", encoding='ISO-8859-1')

row_data = data_items_df.iloc[69:70]
print(row_data["PACKSIZE"].values[0])
qty = int(''.join(re.findall(r'\d+', row_data["PACKSIZE"].values[0])))
uom = ''.join(re.findall(r'[A-Za-z]', row_data["PACKSIZE"].values[0]))
unit = get_key_by_value(uom.lower())
x = master_df[master_df["qty"] == qty]
y = x[x['unit'] == unit]
print(x['ITEMDESC'])