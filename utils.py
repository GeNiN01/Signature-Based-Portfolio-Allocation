import pandas as pd
import numpy as np
import yfinance as yf
from esig import stream2sig, stream2logsig
from scipy import spatial
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.simplefilter("ignore")

def dateparse(d): 
    return pd.Timestamp(d)


def get_historical_close_price(symbols:list, initial_date:str, end_date:str)->pd.DataFrame:
    """
    Parameters
    ----------
    symbols : list
        list of stock
    initial_date : str
        starting data for the data collection
    end_date : str
        ending data for the data collection
    
    Returns
    -------
    stocks : pd.DataFrame
        dataframe with closing price of all the stocks
    """
    stocks = pd.DataFrame()
    for i in symbols:
        tmp_close = yf.download(i,
                          start=initial_date,
                          end=end_date,
                          progress=False)['Close']#['Open', 'Close', 'High', 'Low']
        stocks = pd.concat([stocks, tmp_close], axis=1)
    stocks.columns=symbols
    return stocks

def get_historical_volume(symbols:list, initial_date:str, end_date:str)->pd.DataFrame:
    """
    Parameters
    ----------
    symbols : list
        list of stock
    initial_date : str
        starting data for the data collection
    end_date : str
        ending data for the data collection
    
    Returns
    -------
    stocks : pd.DataFrame
        dataframe with volume of all the stocks
    """
    stocks = pd.DataFrame()
    for i in symbols:
        tmp_close = yf.download(i,
                          start=initial_date,
                          end=end_date,
                          progress=False)['Volume']#['Open', 'Close', 'High', 'Low']
        stocks = pd.concat([stocks, tmp_close], axis=1)
    stocks.columns=symbols
    return stocks

def leadlag_function(X):
    """
    Parameters
    ----------
    X : list
        elements are tuples of the form (time, value)

    Returns
    -------
    l : 
        lead-lag-transformed stream of X
    """
    l=[]
    for j in range(2*(len(X))-1):
        i1=j//2
        i2=j//2
        if j%2!=0:
            i1+=1
        l.append((X[i1][1], X[i2][1]))
    return l

def signature_matrix(df:pd.DataFrame, M:int)->pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        dataframe on which calculate the signature
    M : int
        degree of truncation for the signature

    Returns
    -------
    df_signature_leadlag : pd.DataFrame
        dataframes with the signature computed for each element of df.
    df_logsignature_leadlag : pd.DataFrame
        dataframes with the log-signature computed for each element of df
    """
    variables = df.columns
    leadlag_vector, logleadlag_vector = [], []
    for i, variable in enumerate(variables):
        print('The stock consider is:', variable)
        time_series = df[variable].values
        a = []
        for k,j in enumerate(time_series):
            c = [k, j]
            a.append(c)
        tup = tuple(a)
        leadlag = leadlag_function(tup)
        sig_leadlag = stream2sig(np.array(leadlag), M)
        logsig_leadlag = stream2logsig(np.array(leadlag), M)
        leadlag_vector.append(sig_leadlag)
        logleadlag_vector.append(logsig_leadlag)
        print("Stock remaining: ", df.shape[1] - i)
        print('#####################')
    df_signature_leadlag = pd.DataFrame(leadlag_vector, index=variables)
    df_logsignature_leadlag = pd.DataFrame(logleadlag_vector, index=variables)
    return df_signature_leadlag, df_logsignature_leadlag

def monotonic_transformation(df:pd.DataFrame, monotonic_method:str, a:int=1)->pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        distance matrix dataframe.
    monotonic_method : str
        type of transformation selected
    a : int, optional
        scaling parameter. The default is 1.

    Returns
    -------
    df_transformed : pd.DataFrame
        distance matrix dataframe transofrmed into a similarity matrix dataframe.

    """
    if monotonic_method == 'normalize':
        df_transformed = df/df.max()
    elif monotonic_method == 'fractional':
        df_transformed = 1/(a+df)
    elif monotonic_method == 'exponential':
        df_transformed = np.exp(- df ** a)
    elif monotonic_method == 'arcotangent':
        df_transformed = np.arctan(a * df)
    else:
        print("The monotic trasformation methods allows are: normalize, fractional, exponential, arcotangent")
    return df_transformed

def dtw(s:np.array, t:np.array):
    """
    Parameters
    ----------
    s : np.array
        first array of observation
    t : np.array
        second array of observation

    Returns
    -------
    dtw_value : int 
        value od the Dynamic Time Warping distance
    """
    n, m = len(s), len(t)
    dtw_matrix = np.matrix(np.ones((n, m)) * np.inf)
    dtw_matrix[0, 0] = 0
    for i in range(1, n):
        for j in range(1, m):
            cost = abs(s[i-1] - t[j-1])
            # take last min from a square box
            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix[-1,-1]

def find_similarity_matrix(df:pd.DataFrame, function_type:str, 
                           monotonic_method:str='fractional', a:int=1)->pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        containing the distance matrix or the similarity matrix
    function_type : str
        type of similarity considered. If consider distance metrics, then apply a monotonic transformation to get similarity
    monotonic_method : str, optional
        method used to apply the monotonic function. The default is 'fractional'.
    a : int, optional
       value for the strictly monotonic increasing function  The default is 1.

    Returns
    -------
    sim_mtx : pd.DataFrame
        similarity matrix

    """
    if function_type=='correlation':
        dist_mtx = np.sqrt(0.5 * (1 - df.corr()))
        sim_mtx = monotonic_transformation(dist_mtx, monotonic_method=monotonic_method, a=a)
      
    elif function_type=='DTW': #monotonic transformation needed
        dist_mtx = np.empty([df.shape[1], df.shape[1]])
        for i, stock_1 in enumerate(df.columns):
            for j, stock_2 in enumerate(df.columns):
                dist_mtx[i, j] = dtw(df[stock_1].values, df[stock_2].values)
        dist_mtx = pd.DataFrame(dist_mtx)
        sim_mtx = monotonic_transformation(dist_mtx, monotonic_method=monotonic_method, a=a)
      
    elif function_type=='mutual_information':
        dist_mtx = np.empty([df.shape[1], df.shape[1]])
        for i, stock_1 in enumerate(df.columns):
            for j, stock_2 in enumerate(df.columns):
                dist_mtx[i, j] = mutual_info_regression(df[stock_1].values.reshape(-1, 1), df[stock_2].values.reshape(-1, 1), discrete_features='auto', n_neighbors=df.shape[1] )
        dist_mtx = pd.DataFrame(dist_mtx)
        sim_mtx = monotonic_transformation(dist_mtx, monotonic_method=monotonic_method, a=a)
      
    elif function_type=='euclidean':
        distances = spatial.distance.pdist(df.T.values, metric=function_type)
        dist_mtx = pd.DataFrame(squareform(distances))
        sim_mtx = monotonic_transformation(dist_mtx, monotonic_method=monotonic_method, a=a)
      
    elif function_type=='rbf_kernel':
        kern_mtx = np.empty([df.shape[1], df.shape[1]])
        for i, stock_1 in enumerate(df.columns):
            for j, stock_2 in enumerate(df.columns):
                kern_mtx[i, j] = rbf_kernel(df[stock_1].values.reshape(1, -1), df[stock_2].values.reshape(1, -1), gamma=0.5)
        sim_mtx = pd.DataFrame(kern_mtx)
      
    elif function_type=='cosine':
        distances = spatial.distance.pdist(df.T.values, metric=function_type)
        dist_mtx = pd.DataFrame(squareform(distances))
        sim_mtx = 1 - dist_mtx
      
    elif function_type=='chebyshev':
        distances = spatial.distance.pdist(df.T.values, metric=function_type)
        dist_mtx = pd.DataFrame(squareform(distances))
        sim_mtx = monotonic_transformation(dist_mtx, monotonic_method=monotonic_method, a=a)
      
    elif function_type=='cityblock':
        distances = spatial.distance.pdist(df.T.values, metric=function_type)
        dist_mtx = pd.DataFrame(squareform(distances))
        sim_mtx = monotonic_transformation(dist_mtx, monotonic_method=monotonic_method, a=a)
    else:
        print("Distance not implemented")
      
    return sim_mtx
