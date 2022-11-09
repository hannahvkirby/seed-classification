import pandas as pd
import numpy as np
import sklearn as skl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# get data, file has spaces and tabs as separators
seeds = pd.read_csv("data/seeds_dataset.txt", sep = '\s+|\t| ', 
                    header = None, engine= "python",
                    names = ["area", "perimeter", "compact", "length", 
                    "width", "asymmetry", "groove","seed_type"])

# Use random forest to gain more insight on feature importance.

X_train, X_test, y_train, y_test = train_test_split(seeds.iloc[:, :-1], 
                                seeds.iloc[:, -1:], test_size = 0.3, random_state=1)

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

# subset seeds using only top 3 and top 2 most important features
seeds_subset_3 = seeds[["groove", "area", "perimeter", "seed_type"]]
seeds_subset_2 = seeds[["groove", "area", "seed_type"]]

# for iterating over different seeds subsets
seed_frames = [seeds, seeds_subset_3, seeds_subset_2]
seed_names = ["All features", "Three features", "Two features"]

# perform knn classification for each seeds subset and k 1-10 inclusive
for i,seed in enumerate(seed_frames):
        neighbors = range(1,11)
        print(seed_names[i])
        for n in neighbors:
                X_train, X_test, y_train, y_test = train_test_split(seed.iloc[:, :-1], 
                                seed.iloc[:, -1:], test_size = 0.3, random_state=1)

                knn = KNeighborsClassifier(n_neighbors=n)

                # fit model on training data
                knn.fit(X_train, y_train.values.ravel())

                # predict test values
                y_pred = knn.predict(X_test)

                # compare preds to test data to assess model accuracy
                print("# neighbors: ", n, "\nAccuracy: ",
                      sum(y_pred == y_test.values.ravel())/len(y_pred), "\n")

# highest test accuracy from these combinations is 0.9365 from 
# 2 features (groove and perimeter) and 4 neighbors.


# 93% is decent but let's see if another classifier can more accurately fit the data.

