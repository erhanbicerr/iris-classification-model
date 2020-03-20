from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb 
import pandas

iris = load_iris()
targetNames = list(iris.target_names)
featureNames = list(iris.feature_names)
numSamples, numFeatures = iris.data.shape # since it returns tuple, we can assign them to 2 vars.
print("There are " + str(numSamples) + " samples.")
print("There are " + str(numFeatures) + " features.")
print("These are ; ")
for i in featureNames:
    print("* " + str(i).capitalize())
print("--------------------")
print("There are "  + str(len(targetNames)) + " flower species.")
print("These are ; ")
for i in targetNames:
    print("* " + str(i).capitalize())
    
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)

train = xgb.DMatrix(X_train, label=y_train)
test = xgb.DMatrix(X_test, label=y_test)

param = {
        'max_depth': 3,
        'eta' : 0.3,
        'objective': 'multi:softmax',
        'num_class': 3
      }
epochs = 5 #number of iterations

model = xgb.train(param, train, epochs)

predictions = model.predict(test)
print(predictions)


print(accuracy_score(y_test, predictions))