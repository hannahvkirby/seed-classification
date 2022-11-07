import pandas as pd
import matplotlib.pyplot as plt

# get data, file has spaces and tabs as separators
seeds = pd.read_csv("data/seeds_dataset.txt", sep = '\s+|\t| ', 
                    header = None, engine= "python",
                    names = ["area", "perimeter", "compact", "length", 
                    "width", "asymmetry", "groove", "seed_type"])
                    
# examine data
print(seeds.iloc[0:9])

# get mean and sd from input variables
print("\nMean\n")
print(seeds.iloc[:,0:7].mean())

print("\nStandard Deviation \n")
print(seeds.iloc[:,0:7].std())

# boxplots to visualize above data
seeds.iloc[:,0:7].plot(kind = 'box')
plt.show()
