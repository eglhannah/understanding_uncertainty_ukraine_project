# %%
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl

# %%
data = pd.read_csv("ukraine-damages.csv", delimiter = "|")
a = data['rayon'].isna().sum()
b = data['rayon'].value_counts()
a,b
missing_r = (8720/24269)*100
missing_r
c = data['oblast'].isna().sum()
d = data['oblast'].value_counts()
c,d
missing_o = (8/24269)*100
missing_o
data['rayon'] = data['rayon'].fillna('Missing')
data['oblast'] = data['oblast'].fillna('Unknown')
data

oblasts = data[['oblast']].copy()
oblasts.reset_index(drop=True, inplace=True)
oblasts



# %%
# TRANSITION MATRIX - OBLAST

states_o = list(oblasts["oblast"].unique())
#states_i2

# step 2: compute transition counts/matrix >> adapting code from bach.py
# REMEMBER, it's a normalized matrix

S_o = len(states_o)
#T = len(bach)
tr_counts_o = np.zeros( (S_o, S_o) )

## Compute transition counts:
for row in range(1,len(oblasts["oblast"])): # check two lines down for why we start at 1

    # Current and next tokens:
    x_tm1 = oblasts["oblast"][row-1] # previous state >> this is why we start at 1! bc we are collecting previous state.
    x_t = oblasts["oblast"][row] # current state
    # Determine transition indices:
    index_from = states_o.index(x_tm1)
    index_to = states_o.index(x_t)
    # Update transition counts:
    tr_counts_o[index_from, index_to] += 1

print('Transition Counts:\n', tr_counts_o)

# Sum the transition counts by row >> numpy axis 0 is technically rows, but visually it's columns. "rows" is what we want
sums_o = tr_counts_o.sum(axis=1, keepdims=True)
#print('State proportions: \n')

# Normalize the transition count matrix to get proportions:
tr_pr_o = np.divide(tr_counts_o, sums_o, 
                             out=np.zeros_like(tr_counts_o), 
                             where=sums_o!=0)

#print('Transition Proportions:\n')

oblast_tr_df = pd.DataFrame(np.round(tr_pr_o,2), index=states_o, columns=states_o)
print(oblast_tr_df)

# %%
# oblast projections

# transition matrix: oblast_tr_df
# oblast[0] = Luhanska

# find the index of the starting point

length_o = len(oblast_tr_df.columns)

for i in range(length_o):
    if oblast_tr_df.columns[i] == "Luhanska":
        starting_point_o = i
# initiate the projection matrix and set starting point
init_o = np.zeros(length_o) 
init_o[starting_point_o] = 1
#print(init)

time_windows = [5, 10, 25, 50, 100, 250, 500, 750, 1000, 2000]

init5_o = init_o.copy()
for i in range(5):
   #init5_o = oblast_tr_df @ init5_o
   #init5_o = init5_o @ oblast_tr_df 
   init5_o = init5_o @ tr_pr_o
#print(init2)

init10_o = init_o.copy()
for i in range(10):
   #init10_o = oblast_tr_df@init10_o
   #init10_o = init10_o@oblast_tr_df 
   init10_o = init10_o@tr_pr_o


init25_o = init_o.copy()
for i in range(25):
   #init25_o = oblast_tr_df@init25_o 
   #init25_o = init25_o@oblast_tr_df
   init25_o = init25_o@tr_pr_o  

init50_o = init_o.copy()
for i in range(50):
   #init50_o = oblast_tr_df@init50_o 
   #init50_o = init50_o@oblast_tr_df
   init50_o = init50_o@tr_pr_o

init100_o = init_o.copy()
for i in range(100):
   #init100_o = oblast_tr_df@init100_o 
   #init100_o = init100_o@oblast_tr_df
   init100_o = init100_o@tr_pr_o

init250_o = init_o.copy()
for i in range(250):
   #init250_o = oblast_tr_df@init250_o 
   #init250_o = init250_o@oblast_tr_df
   init250_o = init250_o@tr_pr_o

init500_o = init_o.copy()
for i in range(500):
   #init500_o = oblast_tr_df@init500_o 
   #init500_o = init500_o@oblast_tr_df
   init500_o = init500_o@tr_pr_o

init750_o = init_o.copy()
for i in range(750):
   #init750_o = oblast_tr_df@init750_o 
   #init750_o = init750_o@oblast_tr_df
   init750_o = init750_o@tr_pr_o

init1000_o = init_o.copy()
for i in range(1000):
   #init1000_o = oblast_tr_df@init1000_o 
   #init1000_o = init1000_o@oblast_tr_df
   init1000_o = init1000_o@tr_pr_o

init2000_o = init_o.copy()
for i in range(2000):
   #init2000_o = oblast_tr_df@init2000_o 
   #init2000_o = init2000_o@oblast_tr_df
   init2000_o = init2000_o@tr_pr_o

projections_o = pd.DataFrame({"Initial": init_o,
                            "After 5 strikes": init5_o,
                            "After 10 strikes": init10_o,
                            "After 25 strikes": init25_o,
                            "After 50 strikes": init50_o,
                            "After 100 strikes": init100_o,
                            "After 250 strikes": init250_o,
                            "After 500 strikes": init500_o,
                            "After 750 strikes": init750_o,
                            "After 1000 strikes": init1000_o,
                            "After 2000 strikes": init2000_o})
print(projections_o)

# %%
projections_o["After 2000 strikes"].sum()

# %%
# create GeoDataFrame using the polygons for Ukraine Oblasts

gdf=gpd.read_file("RayonPolygons.shp") # this file is mislabeled, it is really oblasts, not rayons
print(gdf.columns)
gdf["Oblast_City"]=gdf["NAME_1"]
len(gdf["Oblast_City"])



# %%
oblast_name_map=pd.read_csv("UkraineOblastRemap.csv")
oblast_name_map=oblast_name_map[["GeoName", "DamageName"]]
oblast_name_map.info()

# %%
projections_o.columns

# %%
projections_o["oblast_names"]=states_o
projections_o=projections_o.set_index("oblast_names")
projections_o

# %%
projections_long=projections_o.reset_index().melt(
    id_vars="oblast_names",
    var_name="timestep",
    value_name="prob"
).rename(columns={"oblast_names":"oblast"})
projections_long

# %%
oblast_name_map

# %%
gdf['Oblast_City'] = gdf['Oblast_City'].str.strip().str.lower()
oblast_name_map['GeoName'] = oblast_name_map['GeoName'].str.strip().str.lower()


# %%
gdf_m=gdf_m.rename(columns={"DamageName":"oblast"})
gdf_m.info()

# %%
gdf_m=pd.merge(gdf, oblast_name_map, how="left", left_on="Oblast_City", right_on="GeoName")

# %%
merged=gdf_m.merge(projections_long, on="oblast")

# %%
# Merge the oblasts dataframe with the remapped 
# names so they align with those in the shapefile

merged_oblasts=pd.merge(oblasts, oblast_name_map, how="left", left_on="oblast", right_on="DamageName")
merged_oblasts.head()

# %%
gdf=gdf.rename(columns={"NAME_1":"Oblast_City"})
gdf.info()

# %%
merged

# %%
timestep_to_plot = 500 
subset = merged[merged["timestep"] == timestep_to_plot]

subset.plot(column="prob", cmap="Reds", legend=True)
plt.title(f"Strike Probability at {timestep_to_plot}")
plt.show()


# %%
print("Merged GeoDataFrame shape:", merged.shape)
if merged.empty:
    print("Warning: merged GeoDataFrame is empty. Check oblast names for mismatches!")

# %%
timestep_dict={'After 10 strikes':10,
 'After 100 strikes':100,
 'After 1000 strikes':1000,
 'After 2000 strikes':2000,
 'After 25 strikes':25,
 'After 250 strikes':250,
 'After 5 strikes':5,
 'After 50 strikes':50,
 'After 500 strikes':500,
 'After 750 strikes':750,
 'Initial':0}
merged["timestep"]=merged["timestep"].replace(timestep_dict)
merged.head()

# %%



