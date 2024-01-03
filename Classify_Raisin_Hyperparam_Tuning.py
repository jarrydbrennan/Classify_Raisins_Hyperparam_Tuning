# 1. Setup
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

raisins = pd.read_csv('Raisin_Dataset.csv')
raisins.head()

# 2. Create predictor and target variables, X and y
y = raisins['Class']
X = raisins.drop('Class',axis=1)

# 3. Examine the dataset
print("Total number of features", len(X.columns))
print("Total number of samples", len(y))
print("Samples belonging to class '1'", len(raisins[raisins['Class'] == 1])) #raisins.Class.sum()

# 4. Split the data set into training and testing sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 19)

# 5. Create a Decision Tree model
tree = DecisionTreeClassifier()

# 6. Dictionary of parameters for GridSearchCV
parameters = {'max_depth':[3,5,7], 'min_samples_split' : [2,3,4]}

# 7. Create a GridSearchCV model
grid = GridSearchCV(tree,parameters)

#Fit the GridSearchCV model to the training data
grid.fit(X_train,y_train)

# 8. Print the model and hyperparameters obtained by GridSearchCV
print(grid.best_estimator_)

# Print best score
print(grid.best_score_)

# Print the accuracy of the final model on the test data
test_score = grid.score(X_test,y_test)
print(test_score)

# 9. Print a table summarizing the results of GridSearchCV
hyperparam_grid = pd.DataFrame(grid.cv_results_['params'])
grid_scores = pd.DataFrame(grid.cv_results_['mean_test_score'], columns=['Accuracy'])
df=pd.concat([hyperparam_grid,grid_scores], axis = 1)
print(df)

# 10. The logistic regression model
lr = LogisticRegression(solver = 'liblinear', max_iter = 1000)

# 11. Define distributions to choose hyperparameters from
from scipy.stats import uniform
distributions = {'penalty':['l1','l2'], 'C':uniform(loc=0, scale=100)}

# 12. Create a RandomizedSearchCV model
clf = RandomizedSearchCV(lr,distributions,n_iter=8)

# Fit the random search model
clf.fit(X_train,y_train)

# 13. Print best esimatore and best score
print(clf.best_estimator_)
print(clf.best_score_)
#Print a table summarizing the results of RandomSearchCV
hyperparam_random = pd.DataFrame(grid.cv_results_['params'])
random_scores = pd.DataFrame(grid.cv_results_['mean_test_score'], columns=['Accuracy'])
df=pd.concat([hyperparam_random,random_scores], axis = 1)
print(df.sort_values('Accuracy', ascending = False))