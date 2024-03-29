import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# get data, file has spaces and tabs as separators
seeds = pd.read_csv("data/seeds_dataset.txt", sep = '\s+|\t| ', 
                    header = None, engine= "python",
                    names = ["area", "perimeter", "compact", "length", 
                    "width", "asymmetry", "groove", "seed_type"])
                    

# get mean and sd from input variables
print("\nMean")
print(seeds.iloc[:,0:7].mean())

print("\nStandard Deviation")
print(seeds.iloc[:,0:7].std())

# boxplots to visualize above data
seeds.iloc[:,0:7].boxplot()
plt.show()


# boxplots of input variables for each of the three seed types
seeds.groupby("seed_type").boxplot()
plt.show()

# seems like length and width have biggest disparities within
# seed_type. Will continue exploring in analysis.py

# let's also look at the correlation matrix of the predictor variables
X=seeds.iloc[:, :-1]
matrix = pd.DataFrame.cov(X)
sns.heatmap(matrix, annot=True, fmt='g')
plt.show()