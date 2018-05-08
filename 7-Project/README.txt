1) Here is the list of all the libraries required to run the project.
import numpy as np
import pandas as pd
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display 
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, mutual_info_classif, f_regression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.externals.joblib.parallel import cpu_count, Parallel, delayed
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import gc
import tempfile
import os
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.metrics import make_scorer
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from functools import partial
----------------------------------------------------------
2) Most of them are installed in a common anaconda environment except:
nltk, joblib, xgboost, seaborn(sometimes is not installed)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from joblib import load, dump
import xgboost as xgb
----------------------------------------------------------
3) The dataset can be found here: 
https://www.kaggle.com/c/mercari-price-suggestion-challenge/data
I only used the training dataset
---------------------------------------------------------
**NOTE: in the kernel you will find functions to load a file called trainAfterLemmalizer.csv, X_trainMatrix, and X_trainMatrixV2. You can jump these cells, I just created them to avoid to run every time the processes that used to consume much time.

