import pandas as pd

iris = pd.read_csv('elm_iris_experiment.csv')
glass = pd.read_csv('elm_glass_experiment.csv')
wine = pd.read_csv('elm_wine_experiment.csv')

iris_frames = []
iris_grp = iris.groupby('Optimizer')
for name, group in iris_grp:
    group_df = pd.DataFrame(group)
    iris_frames.append(group_df)
    
iris_table = pd.concat(iris_frames, keys=[name for name, _ in iris_grp])
iris_table = iris_table.drop(columns=[iris_table.columns[0], iris_table.columns[1]])

glass_frames = []
glass_grp = glass.groupby('Optimizer')
for name, group in glass_grp:
    group_df = pd.DataFrame(group)
    glass_frames.append(group_df)
    
glass_table = pd.concat(glass_frames, keys=[name for name, _ in glass_grp])
glass_table = glass_table.drop(columns=[glass_table.columns[0], glass_table.columns[1]])

wine_frames = []
wine_grp = wine.groupby('Optimizer')
for name, group in wine_grp:
    group_df = pd.DataFrame(group)
    wine_frames.append(group_df)
    
wine_table = pd.concat(wine_frames, keys=[name for name, _ in wine_grp])
wine_table = wine_table.drop(columns=[wine_table.columns[0], wine_table.columns[1]])


wine = wine.drop(columns=[wine.columns[0], wine.columns[1]])
glass = glass.drop(columns=[glass.columns[0], glass.columns[1]])
iris = iris.drop(columns=[iris.columns[0], iris.columns[1]])