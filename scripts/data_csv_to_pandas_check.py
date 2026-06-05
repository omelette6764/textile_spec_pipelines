import pandas as pd

df = pd.read_csv("BTVIZ_2025-11-03_effortful_swallow_and_masako_maneuver_and_water.csv")
print("Unique Environment values:", df["Environment"].dropna().unique()[:10])
print("Unique Activity values:", df["Activity"].dropna().unique()[:20])
