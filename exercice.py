#from __future__ import print_function
import pandas as pd 
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from itertools import product
import os
import subprocess
from sklearn.tree import export_graphviz
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
# to print all elements of an array\ndarray\...
#np.set_printoptions(threshold=np.nan)  


data = pd.read_csv("Data.csv" , sep =";") 


fname_in = 'Data.csv'
fname_out = 'your_finished_csv_file.csv'
with open(fname_in, 'rt', encoding="ascii") as fin, open(fname_out, 'wt', encoding="ascii") as fout:
    reader = csv.reader(fin, delimiter=';')
    writer = csv.writer(fout, delimiter=';')
    for row in reader:
        writer.writerow(row[2:])

data_timeRemoved = pd.read_csv("your_finished_csv_file_points.csv" , sep =";")



features = ['Variable1', 'Variable2', 'Variable3', 'Variable4', 'Variable5', 'Variable6', 'Variable7', 'Variable8', 'Variable9', 'Variable10', 'Variable11', 'Variable12', 'Variable13', 'Variable15', 'Variable16', 'Variable17', 'Variable18', 'Variable19', 'Variable20', 'Variable21', 'Variable22', 'Variable23', 'Variable24', 'Variable25', 'Variable26', 'Variable27', 'Variable28', 'Variable29', 'Variable30', 'Variable31', 'Variable32', 'Variable33', 'Variable34', 'Variable35', 'Variable36', 'Variable37', 'Variable38', 'Variable39', 'Variable40', 'Variable41', 'Variable42', 'Variable43', 'Energy', 'Variable44', 'Variable45', 'Variable46', 'Variable47'] 
x = data_timeRemoved.loc[:, features].values



x = StandardScaler().fit_transform(x)
OriginalDataDf = pd.DataFrame(data = x, columns = features)
#print(OriginalDataDf.head())


pca = PCA(n_components=4)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents , columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4'])
#print(principalDf)


###---------------###
###   plotting    ###
###---------------###

pca_2 = PCA(n_components=2)
principalComponents_2 = pca_2.fit_transform(x)
principalDf_2 = pd.DataFrame(data = principalComponents_2 , columns = ['principal component 1', 'principal component 2'])
"""
principalDf_2.plot(kind='scatter',x='principal component 1',y='principal component 2',color='red')
#plt.show()
"""

###-------------------------------------------------------------------------###
###          Applying Kmeans to Indiv Proj Matrix (with 4 PCA's)            ###
###-------------------------------------------------------------------------###

# Convert DataFrame to matrix
mat = principalDf.as_matrix()
# Using sklearn.KMeans
km = KMeans(n_clusters=4)
km.fit(mat)
# Get cluster assignment labels
labels = km.labels_                 
#print(labels)  #labels is an ndarray

# Format results as a DataFrame
results = pd.DataFrame([principalDf.index,labels]).T
#print(results)

principalDf['cluster'] = labels.tolist()
#print(principalDf)

###-------------------------------------------------------------------------###
###              Merging the Cluster column with original data              ###
###-------------------------------------------------------------------------###

OriginalDataDf['cluster'] = labels.tolist()
#print(OriginalDataDf)

###-------------------------------------------------------------------------###
###              plotting the clusters with different colors                ###
###-------------------------------------------------------------------------###

finalDf = pd.concat([principalDf_2, OriginalDataDf[['cluster']]], axis = 1)
#print(finalDf)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)

targets = [0, 1, 2, 3]
colors = ['r', 'g', 'b', 'y']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['cluster'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
#plt.show()


###-------------------------------------------------------------------------###
###          Applying Decision Tree on the new data and plotting it         ###
###                                                                         ###
###         principal component 1 , principal component 2 , cluster         ###
###-------------------------------------------------------------------------###

principalDf_2['cluster'] = labels.tolist()   # principal component 1 , principal component 2 , cluster

features1 = list(principalDf_2.columns[:2]) # ['principal component 1', 'principal component 2']

X = principalDf_2[features1]    # X is a dataframe
y = principalDf_2["cluster"]
clf_dt = DecisionTreeClassifier().fit(X, y)

# Plotting decision regions

X = X.as_matrix() # X is an ndarray

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

Z = clf_dt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

plt.scatter(X[:, 0], X[:, 1], c=y, s=15, edgecolor='k')
plt.suptitle('Decision surface of a decision tree using PC1 & PC2', fontsize = 15)
plt.show() 

###-------------------------------------------------------------------------###
###                   Applying Decision Tree on the whole data              ###
###                                                                         ###
###         OriginalDataDf: Variable1-...Variable47 , cluster               ###
###-------------------------------------------------------------------------###

OriginalDataDf['cluster'] = labels.tolist()  

X = OriginalDataDf[features]    # X is a dataframe
y = OriginalDataDf["cluster"]
clf_dt = DecisionTreeClassifier().fit(X, y)

def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f, feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to produce visualization")


#visualize_tree(clf_dt, features)

###-------------------------------------------------------------------------###
###                 for each cluster apply LR "Energy"                      ###
###-------------------------------------------------------------------------###

# split OriginalDataDf into 4 dataframes by cluster value
df0 = OriginalDataDf[OriginalDataDf['cluster'] == 0]
df1 = OriginalDataDf[OriginalDataDf['cluster'] == 1]
df2 = OriginalDataDf[OriginalDataDf['cluster'] == 2]
df3 = OriginalDataDf[OriginalDataDf['cluster'] == 3]

dataframes = [df0, df1, df2, df3]
dfNumber = 0

for dfn in dataframes:
	count = dfn.shape[0]
	nb = int(count * 0.2)
	print(count , nb)

	# Split the data into training/testing sets
	dfn_X_train = pd.DataFrame()
	dfn_X_train = dfn[:-nb]
	del dfn_X_train['Energy']

	#### print('dfn_X_train' + str(dfNumber))
	#### print(dfn_X_train.head())
	
	dfn_X_test = pd.DataFrame()
	dfn_X_test = dfn[-nb:]
	del dfn_X_test['Energy']

	#### print('dfn_X_test' + str(dfNumber))
	#### print(dfn_X_test.head())

	# Split the targets into training/testing sets
	dfn_y_train = pd.DataFrame()
	dfn_y_train = dfn.Energy[:-nb]
	dfn_y_test = pd.DataFrame()
	dfn_y_test = dfn.Energy[-nb:]

	# Create linear regression object
	regr = linear_model.LinearRegression()

	# Train the model using the training sets
	regr.fit(dfn_X_train, dfn_y_train)

	# Make predictions using the testing set
	dfn_y_pred = []
	dfn_y_pred = regr.predict(dfn_X_test)

	# The coefficients
	#print('Coefficients: \n', regr.coef_)

	if dfNumber == 0 :
		plt.subplot(221)

	if dfNumber == 1 :
		plt.subplot(222)

	if dfNumber == 2 :
		plt.subplot(223)

	if dfNumber == 3 :
		plt.subplot(224)

	plt.title('Cluster: ' + str(dfNumber))
	plt.scatter(dfn_y_test, dfn_y_pred,  color='red', s = 4)
	plt.plot(dfn_y_pred, dfn_y_pred, color='blue', linewidth=0.5)

	plt.xticks(())
	plt.yticks(())
	plt.suptitle('Linear regression on each cluster', fontsize = 20)
	plt.tight_layout(h_pad=1.5, w_pad=1.5, pad=3.5)

	plt.xlabel('Real Energy', fontsize = 8)
	plt.ylabel('Predicted Energy', fontsize = 8)
	dfNumber = dfNumber + 1


plt.show()

##########################################################
####                                                  ####
#### LR on the original data frame  OriginalDataDf    ####
####                                                  ####
##########################################################
dfn = pd.DataFrame()
dfn = OriginalDataDf
#print(dfn['Energy'])
count = dfn.shape[0]
nb = int(count * 0.2)
print(count , nb)

# Split the data into training/testing sets
dfn_X_train = pd.DataFrame()
dfn_X_train = dfn[:-nb]
del dfn_X_train['Energy']
print(dfn_X_train)

dfn_X_test = pd.DataFrame()
dfn_X_test = dfn[-nb:]
del dfn_X_test['Energy']

#### print('dfn_X_test' + str(dfNumber))
#### print(dfn_X_test.head())

# Split the targets into training/testing sets
dfn_y_train = pd.DataFrame()
dfn_y_train = dfn.Energy[:-nb]
dfn_y_test = pd.DataFrame()
dfn_y_test = dfn.Energy[-nb:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(dfn_X_train, dfn_y_train)
print(dfn_y_train)
# Make predictions using the testing set
dfn_y_pred = []
dfn_y_pred = regr.predict(dfn_X_test)

print('Variance score: %.2f' % r2_score(dfn_y_test, dfn_y_pred))

plt.scatter(dfn_y_test, dfn_y_pred,  color='black')
#plt.plot(dfn_y_test, dfn_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
#print(OriginalDataDf['Energy'])