import pandas as pd
import numpy as np
import sklearn as skl

# get data, file has spaces and tabs as separators
seeds = pd.read_csv("../data/seeds_dataset.txt", sep = '\s+|\t| ', 
                    header = None, engine= "python",
                    names = ["area", "perimeter", "compact", "length", 
                    "width", "asymmetry", "groove", "seed_type"])
                    
# examine data
print(seeds.iloc[0:9])

# get mean and sd from input variables
print("Mean\n")
print(seeds.iloc[:,0:7].mean())

print("\n Standard Deviation \n")
print(seeds.iloc[:,0:7].std())

