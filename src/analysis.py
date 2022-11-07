import pandas as pd
import numpy as np
import sklearn as skl
import matplotlib.pyplot as plt

# get data, file has spaces and tabs as separators
seeds = pd.read_csv("data/seeds_dataset.txt", sep = '\s+|\t| ', 
                    header = None, engine= "python",
                    names = ["area", "perimeter", "compact", "length", 
                    "width", "asymmetry", "groove", "seed_type"])

