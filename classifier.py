import numpy as np
import pandas as pd

# IMPORT SCIKIT-LEARN MODULES
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

# IMPORT CLASSIFIER MODULES
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# IMPORT MODULES FOR PLOTTING
import seaborn as sns
from matplotlib import pyplot as plt

class Classifier:
    """Container for analysing the dataset using different classification models."""
    def __init__(self, estimator, fname: str, **kwargs):
        
        # initialise the classifier
        self.estimator = estimator(**kwargs)
        
        # import csv as a dataframe and tag rows containing '-' as missing values
        self.df = pd.read_csv(fname, na_values=['-'])
        
    def clean_up(self):
        """Remove rows with mising data and invalid data."""
        
        # Drop missing values and reset index
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        
        # Drop rows that do not satisfy the requirement 2 as above
        for index in self.df.index:
            total_valency = int(self.df['v(A)'][index]) + int(self.df['v(B)'][index])
            if total_valency != 6:
                self.df.drop(index, inplace=True)
            if not (0.82 < float(self.df['tG'][index]) < 1.1) or not (0.414 < float(self.df['μ'][index]) < 0.732):
                self.df.drop(index, inplace=True)
                
        self.df.reset_index(drop=True, inplace=True)
        
    def run_classifier(self):
        """Sets the feature and target columns, performs the training/testing split, and fits the classifier."""
        
        # set feature columns as every column besides the 'Compound', 'A', 'B', and 'Lowest distortion' columns
        self.X = self.df.drop(['Compound', 'A', 'B', 'Lowest distortion'], axis=1)
        
        # drop highly correlated features
        self.X = self.X.drop(['v(B)', 'r(AVI)(Å)'], axis=1)
        
        # feature scaling for X using min-max scaling
        self.X = MinMaxScaler().fit_transform(self.X)
        
        # set target column as the 'Lowest distortion' column
        self.y = self.df['Lowest distortion']
        
        # training and testing split with random ratio
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=0.2, random_state=1)
        
        # fit training set to estimator
        self.estimator.fit(self.X_train, self.y_train)
        
        # generate prediction set
        self.y_pred = self.estimator.predict(self.X_test)
        
    def correlation(self):
        """Plots a heatmap of the correlation matrix."""
        correlation_matrix = self.df.corr()
        correlation_heatmap = sns.heatmap(correlation_matrix, cmap="RdPu", annot=True, fmt=".2f", cbar=False)
        return correlation_heatmap
        
    def score(self):
        """Returns the accuracy score of the latest prediction."""
        return accuracy_score(self.y_test, self.y_pred)
    
    def grid(self, **kwargs):
        """Runs the GridSearch cross-validation algorithm to find the best parameters and CV score for the classifier."""
        self.param_grid = {
            'bootstrap': [True],
            'max_depth': [10, 40, 70, 100],
            'max_features': [1,2],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 5, 10],
            'n_estimators': [200, 500, 1000, 2000]
        }
        self.grid_search = GridSearchCV(
            estimator = self.estimator, 
            param_grid = self.param_grid, 
            cv = 3,
            n_jobs = -1, 
            verbose = 2, 
            scoring = 'accuracy'
        )
        self.grid_search.fit(self.X_train, self.y_train)
        return self.grid_search.best_params_, self.grid_search.score(self.X_test, self.y_test)
        

if __name__ == "__main__":

    # Initialise class
    classifier = Classifier(RandomForestClassifier, 'Crystal_structure.csv')
    
    # Run preprocessing
    classifier.clean_up()
    
    # Run model
    classifier.run_classifier()
    
    # Print accuracy score
    print(f"Accuracy score of the classifier is: {classifier.score()}")
    
    # Print correlation matrix
    print(classifier.correlation())
    
    # Run GridSearch CV algorithm to find best parameters for the model
    best_params, new_score = classifier.grid()

    print(f'The best parameters for the classifier are {best_params}\n which yields an accuracy score of {new_score}')