import pandas as pd
import numpy as np
import networkx as nx
from esig import stream2sig
from network_function import network_metrics
from utils import leadlag_function, find_similarity_matrix 
from pypfopt.efficient_frontier import EfficientFrontier
import cvxpy as cp

def find_baseline_weight(df_returns:pd.DataFrame, stocks_list:list, risk_free:float, 
                         rebalancing_day:int, type_strategy:str, short=False):
    """
    Parameters
    ----------
    df_returns : pd.DataFrame
        dataframe with returns.
    stocks_list : list
        list of stocks consdier for the portfolio (list of string)
    risk_free : float
        risk free rate 
    rebalancing_day : int
        number of days needed before optimize and find new weight (No rebalancing = df_returns.shape[0])
    type_strategy : str
        type of strategy selected for find the optimal weight
    short : Booleans, optional
        Short sell allow or not. The default is False.

    Returns
    -------
    df_weights : pd.DataFrame
        dataframe containing the optimal weight given the selected strategy.
    rebalancing_day_list : list
        list of day where the rebalancing happen.

    """
    if short:
        weight_bound = (-1,1)
    else: 
        weight_bound = (0,1)
    count = rebalancing_day
    rebalancing_day_list = []
    df_firstrow = pd.DataFrame(df_returns[stocks_list].iloc[0]).T
    df_selected = df_returns[stocks_list].iloc[1:]
    df_weights = pd.DataFrame(columns=df_selected.columns)
    for index, value in df_selected.iterrows():
        if count == rebalancing_day:
            df_combined = pd.concat([df_firstrow, df_selected.loc[:index]], axis=0)
            mu = df_combined.mean()
            sigma = df_combined.cov()
            # Portfolio Optimizaiton
            ef = EfficientFrontier(mu, sigma, weight_bounds=weight_bound,solver=cp.ECOS)
            if type_strategy == "mean_variance": 
                weights_ret = ef.max_quadratic_utility(risk_aversion=2)
            elif type_strategy == "max_sharpe": 
                weights_ret = ef.max_sharpe(risk_free_rate=risk_free)
            elif type_strategy == "minimum_variance": 
                weights_ret = ef.min_volatility()
            else: 
                return print(f"The {type_strategy} portfolio strategy is not implement yet.")
            # Find rebalancing day
            rebalancing_day_list.append(index)
            # Set to zero the count for the rebalancing day
            count = 0
        # Create a dataframe for the portfolio weight
        df_weights = pd.concat([df_weights,pd.DataFrame(list(weights_ret.values()),index=df_selected.columns).T],axis=0)
        count+=1
    return pd.DataFrame(df_weights.values, index=df_selected.index, columns=df_selected.columns).round(3),rebalancing_day_list

def network_sigma(df:pd.DataFrame, threshold:float, T:int)->pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        dataframes containing the returns.
    threshold : float
        threshold used to computed the filtering of the correlation matrix.
    T : int, optional
        number of observation.

    Returns
    -------
    Sigma : pd.DataFrame
        dataframe with the matrix used for substitute the correlation matrix for the 
        portfolio optimization using the network approach.
    """
    c_tau_ = np.round((np.exp(2*threshold/np.sqrt(T-3))-1)/(np.exp(2*threshold/np.sqrt(T-3))+1), 6)
    # Filter the correlation matrix
    corr_matrix = df.corr()
    adj_mtx = np.where(np.abs(corr_matrix)>c_tau_, np.abs(corr_matrix), 0)
    G_threshold=nx.from_numpy_array(adj_mtx)
    # Compute the network statistics
    clustering_coefficent = network_metrics(G_threshold, "Clustering Coefficient")
    H = np.matrix(np.zeros((df.shape[1], df.shape[1])))
    for i in clustering_coefficent.keys():
        for j in clustering_coefficent.keys():
            if i ==j: 
                H[i,j] = 1
            else:
                H[i,j] = clustering_coefficent[i]*clustering_coefficent[j]
    H = pd.DataFrame(H)
    Delta = pd.DataFrame(np.diag(df.std()/np.sqrt(df.var().sum())))
    mul = np.matmul(np.matmul(Delta.T, H),Delta)
    Sigma = pd.DataFrame(mul.values, columns=df.columns, index=df.columns) 
    return Sigma

def find_portoflio_network_weight(df_returns:pd.DataFrame, stocks_list:list, risk_free:float, 
                                  theta:float, rebalancing_day:int, type_strategy:str, short=False):
    """
    Parameters
    ----------
    df_returns : pd.DataFrame
        dataframes containing the returns.
    stocks_list : list
        list of stocks consdier for the portfolio (list of string)
    risk_free : float
        risk free rate 
    theta : float
        threshold for filtering the correlation matrix.
    rebalancing_day : int
        number of days needed before optimize and find new weight (No rebalancing = df_returns.shape[0])
    type_strategy : str
        type of strategy selected for find the optimal weight
    short : Booleans, optional
        Short sell allow or not. The default is False.

    Returns
    -------
    df_weights : pd.DataFrame
        dataframe containing the optimal weight given the selected strategy.
    rebalancing_day_list : list
        list of day where the rebalancing happen.

    """
    if short: 
        weight_bound = (-1,1)
    else:
        weight_bound = (0,1)
    count = rebalancing_day
    rebalancing_day_list = []
    df_firstrow = pd.DataFrame(df_returns[stocks_list].iloc[0]).T
    df_selected = df_returns[stocks_list].iloc[1:]
    df_weights = pd.DataFrame(columns=df_selected.columns)
    for index, value in df_selected.iterrows(): 
        if count == rebalancing_day:
            df_combined = pd.concat([df_firstrow, df_selected.loc[:index]], axis=0)
            mu = df_combined.mean()
            sigma = network_sigma(df=df_combined, threshold=theta, T=T_)
            # Portfolio Optimizaiton
            ef = EfficientFrontier(mu, sigma, weight_bounds=weight_bound,solver=cp.ECOS)#, verbose=True)
            if type_strategy == "mean_variance": 
                weights_ret = ef.max_quadratic_utility(risk_aversion=2)
            elif type_strategy == "max_sharpe": 
                weights_ret = ef.max_sharpe(risk_free_rate=risk_free)
            elif type_strategy == "minimum_variance": 
                weights_ret = ef.min_volatility()
            else: 
                return print(f"The {type_strategy} portfolio strategy is not implement yet.")
            # Find rebalancing day
            rebalancing_day_list.append(index)
            # Set to zero the count for the rebalancing day
            count = 0
        # Create a dataframe for the portfolio weight
        df_weights = pd.concat([df_weights,pd.DataFrame(list(weights_ret.values()),index=df_selected.columns).T],axis=0)
        count+=1
    return pd.DataFrame(df_weights.values, index=df_selected.index, columns=df_selected.columns).round(3),rebalancing_day_list

def signature_matrix_bis(df:pd.DataFrame, M:int)->pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        dataframe on which calculate the signature
    M : int
        degree of truncation for the signature.

    Returns
    -------
    df_signature_leadlag : pd.DataFrame
        dataframe with the signature computed using the Lead-Lag transformation

    """
    variables = df.columns
    leadlag_vector = []
    for i, variable in enumerate(variables):
        time_series = df[variable].values
        a = []
        for k,j in enumerate(time_series):
            c = [k, j]
            a.append(c)
        tup = tuple(a)
        leadlag = leadlag_function(tup)
        sig_leadlag = stream2sig(np.array(leadlag), M)
        leadlag_vector.append(sig_leadlag)
    df_signature_leadlag = pd.DataFrame(leadlag_vector, index=variables)
    return df_signature_leadlag

def network_sigma_signature(df:pd.DataFrame, threshold:float, T:int)->pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        dataframes containing the returns.
    threshold : float
        threshold used to computed the filtering of the correlation matrix.
    T : int, optional
        number of observation.

    Returns
    -------
    Sigma : pd.DataFrame
        dataframe with the matrix used for substitute the correlation matrix for the 
        portfolio optimization using the signature approach.
    """
    c_tau_ = np.round((np.exp(2*threshold/np.sqrt(T-3))-1)/(np.exp(2*threshold/np.sqrt(T-3))+1), 6)
    # Find the Signature
    df_signature_leadlag = signature_matrix_bis(df, M=5)
    df_signature_leadlag = df_signature_leadlag.T
    sim_mtx = find_similarity_matrix(df_signature_leadlag, function_type='euclidean')
    sim_mtx = sim_mtx.set_index(df.columns)
    sim_mtx.columns = df.columns
    # Filter the signature-based similarity matrix
    adj_mtx = np.where(np.abs(sim_mtx)>c_tau_, np.abs(sim_mtx), 0)
    G_threshold=nx.from_numpy_array(adj_mtx)
    # Compute the network statistics
    clustering_coefficent = network_metrics(G_threshold, "Clustering Coefficient")
    H = np.matrix(np.zeros((df.shape[1], df.shape[1])))
    for i in clustering_coefficent.keys():
        for j in clustering_coefficent.keys():
            if i ==j: 
                H[i,j] = 1
            else: 
                H[i,j] = clustering_coefficent[i]*clustering_coefficent[j]
    H = pd.DataFrame(H)
    Delta = pd.DataFrame(np.diag(df.std()/np.sqrt(df.var().sum())))
    mul = np.matmul(np.matmul(Delta.T, H),Delta)
    Sigma = pd.DataFrame(mul.values, columns=df.columns, index=df.columns) 
    return Sigma

def find_portoflio_sig_network_weight(df_returns:pd.DataFrame, stocks_list:list, risk_free:float,
                                      theta:float, rebalancing_day:int, type_strategy:str, short=False):
    """
    Parameters
    ----------
    df_returns : pd.DataFrame
        dataframes containing the returns.
    stocks_list : list
        list of stocks consdier for the portfolio (list of string)
    risk_free : float
        risk free rate 
    theta : float
        threshold for filtering the correlation matrix.
    rebalancing_day : int
        number of days needed before optimize and find new weight (No rebalancing = df_returns.shape[0])
    type_strategy : str
        type of strategy selected for find the optimal weight
    short : Booleans, optional
        Short sell allow or not. The default is False.

    Returns
    -------
    df_weights : pd.DataFrame
        dataframe containing the optimal weight given the selected strategy.
    rebalancing_day_list : list
        list of day where the rebalancing happen.

    """
    if short: 
        weight_bound = (-1,1)
    else: 
        weight_bound = (0,1)
    count = rebalancing_day
    rebalancing_day_list = []
    df_firstrow = pd.DataFrame(df_returns[stocks_list].iloc[0]).T
    df_selected = df_returns[stocks_list].iloc[1:]
    df_weights = pd.DataFrame(columns=df_selected.columns)
    for index, value in df_selected.iterrows():
        if count == rebalancing_day:
            df_combined = pd.concat([df_firstrow, df_selected.loc[:index]], axis=0)
            mu = df_combined.mean()
            sigma = network_sigma_signature(df=df_combined, threshold=theta, T=T_)
            # Portfolio Optimizaiton
            ef = EfficientFrontier(mu, sigma, weight_bounds=weight_bound,solver=cp.ECOS)#, verbose=True)
            if type_strategy == "mean_variance": 
                weights_ret = ef.max_quadratic_utility(risk_aversion=2)
            elif type_strategy == "max_sharpe": 
                weights_ret = ef.max_sharpe(risk_free_rate=risk_free)
            elif type_strategy == "minimum_variance": 
                weights_ret = ef.min_volatility()
            else: 
                return print(f"The {type_strategy} portfolio strategy is not implement yet.")
            # Find rebalancing day
            rebalancing_day_list.append(index)
            # Set to zero the count for the rebalancing day
            count = 0
        # Create a dataframe for the portfolio weight
        df_weights = pd.concat([df_weights,pd.DataFrame(list(weights_ret.values()),index=df_selected.columns).T],axis=0)
        count+=1
    return pd.DataFrame(df_weights.values, index=df_selected.index, columns=df_selected.columns).round(3),rebalancing_day_list

def find_strategy_name(i:int)->str:
    """
    Parameters
    ----------
    i : int
        position of the dataframe in a list

    Returns
    -------
    strategy_name : str
        strategy name used for the analysis

    """
    if i == 0: 
        return "GMV"
    elif i == 1: 
        return "MV"
    elif i == 2: 
        return "MS"
    elif i == 3: 
        return "EW"
    elif i == 4:
        return "NGMV"
    elif i == 5:
        return "NMS"
    elif i == 6:
        return "NMV"
    elif i == 7:
        return "Sig_NMGV"
    elif i == 8:
        return "Sig_NMS"
    else:
        return "Sig_NMV"

def compute_main_statistics(dfs:list, number_asset:list, tau_list:list, rf:float=0.0):
    """
    Parameters
    ----------
    dfs : list
        list of dataframes where each dataframe contains the portfolio returns.
    number_asset : list
        list of string, each element represents the number of asset consider for each portfolio.
    tau_list : TYPE
        list of float, each elements is the threshold used for filter the correlation matrix or 
        the signature-based similarity matrix
    rf : float, optional
        risk free rate. The default is 0.0.

    Returns
    -------
    df_statistics : pd.DataFrame
        dataframe containing the evaluation metric used in the analysis.

    """
    main_stat = []
    for i, df in enumerate(dfs):
        name = find_strategy_name(i)
        if type(df) is dict:
            for t in tau_list:
                for asset in number_asset:
                    mean = (((1+df[t][asset].mean())**252) -1) *100
                    std = (df[t][asset].std()*np.sqrt(252)) *100
                    kurt = df[t][asset].kurt() - 3
                    skw = df[t][asset].skew()
                    cum_ret = df[t][asset].cumsum()[-1] *100
                    sharpe = ((mean - rf)/std) #*100
                    drowdowns = ((1+df[t][asset]).cumprod() - (1+df[t][asset]).cumprod().cummax())/(1+df[t][asset]).cumprod().cummax()
                    max_drow = drowdowns.min()*100
                    sort_ret = np.sort(df[t][asset])
                    n_sample = len(sort_ret)
                    cvar_idx = int((1 - 0.95) * n_sample)
                    cvar = np.mean(sort_ret[:cvar_idx]) * 100
                    c = [name, t, asset, mean, std, kurt, skw, cum_ret, sharpe, max_drow, cvar]#, ir]
                    main_stat.append(c)
        else:
            for asset in number_asset:
                mean = (((1+df[asset].mean())**252) -1) * 100
                std = (df[asset].std()*np.sqrt(252)) *100
                kurt = df[asset].kurt() - 3
                skw = df[asset].skew()
                cum_ret = df[asset].cumsum()[-1] *100
                sharpe = ((mean - rf)/std) #*100
                drowdowns = ((1+df[asset]).cumprod() - (1+df[asset]).cumprod().cummax())/(1+df[asset]).cumprod().cummax()
                max_drow = drowdowns.min()*100
                sort_ret = np.sort(df[asset])
                n_sample = len(sort_ret)
                cvar_idx = int((1 - 0.95) * n_sample)
                cvar = np.mean(sort_ret[:cvar_idx]) * 100
                c = [name, 0, asset, mean, std, kurt, skw, cum_ret, sharpe, max_drow, cvar]#, ir]
                main_stat.append(c)
    df_statistics = pd.DataFrame(main_stat, columns=["Strategy", "Theta", "Number Asset", "Mean Return", "Std Return", "Excess kurtosis", 
                                                      "Skewness", "Cumulative Return", "Sharpe Ratio", "Max Drowdown", "CVaR 95%"])#, "Information Ratio"])
    return df_statistics
      