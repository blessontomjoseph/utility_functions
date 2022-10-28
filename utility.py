import scipy
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

from sklearn.covariance import EllipticEnvelope



def qq_plot(df, variable):
    # function to plot a histogram and a Q-Q plot
    # side by side, for a certain variable
    
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    df[variable].hist()
    plt.subplot(1, 2, 2)
    stats.probplot(df[variable], dist="norm", plot=plt)
    plt.show()
    
    

def elliptic_envelope(data,plot=False,features=None):
#outlier detection using elliptic envelope algorithm
#input is all the columns of the data wea look at the outliers as a whole in here
# if plot=True max-2 features to be given as a list

    elpenv = EllipticEnvelope(contamination=0.025, random_state=1)
    # Returns 1 of inliers, -1 for outliers
    elpenv.fit(data)
    preds=elpenv.predict(data)
    # Extract outliers
    outlier_index = np.where(pred==-1)
   
    if plot:
        outlier_val_feat_1 = data[features[0]][outlier_index]
        outlier_val_feat_2 = data[features[1]][outlier_index]
        sns.scatterplot(data=data,x=features[0],y=features[1]);
        sns.scatterplot(x=outlier__val_feat_1, y=outlier_val_feat_2, color='r');
    return outlier_index




from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



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
















