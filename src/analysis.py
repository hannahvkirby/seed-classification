import pandas as pd
import numpy as np
import sklearn as skl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# get data, file has spaces and tabs as separators
seeds = pd.read_csv("data/seeds_dataset.txt", sep = '\s+|\t| ', 
                    header = None, engine= "python",
                    names = ["area", "perimeter", "compact", "length", 
                    "width", "asymmetry", "groove","seed_type"])

# Use random forest to gain more insight on feature importance.

X_train, X_test, y_train, y_test = train_test_split(seeds.iloc[:, :-1], seeds.iloc[:, -1:], test_size = 0.3, random_state=1)

# feature scaling
sc = skl.preprocessing.StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# training / test dataframe
cols = ["area", "perimeter", "compact", "length", 
        "width", "asymmetry", "groove"]
X_test_std = pd.DataFrame(X_test_std, columns=cols)

forest = RandomForestClassifier(n_estimators=500,
                                random_state=1)

# train the mode
forest.fit(X_train_std, y_train.values.ravel())
importances = forest.feature_importances_

# sort the feature importance in descending order
sorted_indices = np.argsort(importances)[::-1]
 
feat_labels = seeds.columns[:-1]
print(feat_labels)
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[sorted_indices[f]],
                            importances[sorted_indices[f]]))
# as observed when exploring the data, area and perimeter are 
# both high on the list of importance. groove also appears to
# have a strong influence on seed_type

# perform knn classification using top 3 important variables

