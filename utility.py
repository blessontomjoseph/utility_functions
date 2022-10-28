import scipy
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")


def qq_plot(df, variable):
    # function to plot a histogram and a Q-Q plot
    # side by side, for a certain variable
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    df[variable].hist()
    plt.subplot(1, 2, 2)
    stats.probplot(df[variable], dist="norm", plot=plt)
    plt.show()
    
    
    
    
    
# all the below algorithms return 1 for inliers and -1 for outliers




# elliptic envelope initialization
from sklearn.covariance import EllipticEnvelope
elpenv = EllipticEnvelope(contamination=0.025, random_state=1)

def elliptic_envelope(data,plot=False,features=None,elpenv):
#outlier detection using elliptic envelope algorithm
#input is all the columns of the data wea look at the outliers as a whole in here
# if plot=True max-2 features to be given as a list

    elpenv.fit(data)
    preds=elpenv.predict(data)
    outlier_index = np.where(pred==-1)
   
    if plot:
        outlier_val_feat_1 = data[features[0]][outlier_index]
        outlier_val_feat_2 = data[features[1]][outlier_index]
        sns.scatterplot(data=data,x=features[0],y=features[1]);
        sns.scatterplot(x=outlier__val_feat_1, y=outlier_val_feat_2, color='r');
    return outlier_index




# isolation_forest initialization referance
from sklearn.ensemble import IsolationForest
iforest = IsolationForest(n_estimators=100, max_samples='auto', 
                          contamination=0.05, max_features=1.0, 
                          bootstrap=False, n_jobs=-1, random_state=1)

def isolation_forest(data,iforest)
    # iforest in an instance of isolation_forest
    # Returns 1 of inliers, -1 for outliers
    pred = iforest.fit_predict(data)
    outlier_index = np.where(pred==-1)
    outlier_values = data[outlier_index]
    return outlier_index,outlier_values




# lof initialization referance
from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor(n_neighbors=20, algorithm='auto',
                         metric='minkowski', contamination=0.04,
                         novelty=False, n_jobs=-1)

class Lof:
    def __init__(self,data,lof):
        self.lof=lof
        self.data=data
        
    def outlier(self):
        pred = self.lof.fit_predict(self.data)
        outlier_index = np.where(pred==-1)
        outlier_values = data.iloc[outlier_index]
        return outlier_index,outlier_values
    
    def novalty(self,val_data):
        self.lof.fit(self.data)
        preds=self.lof.predict(val_data)
        outlier_index = np.where(pred==-1)
        outlier_values = val_data.iloc[outlier_index]
        return outlier_index,outlier_values
    













