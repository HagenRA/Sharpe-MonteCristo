# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 14:10:38 2021
Adapted for top performers 31st December

@author: Hagen Arhelger
"""

# Dependencies
import requests, os, json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import json_normalize
from datetime import datetime, timedelta
from scipy.optimize import minimize
from discord import Webhook, RequestsWebhookAdapter, File
from instabot import Bot
from tqdm import tqdm
from binance.client import Client
#%
webhook = Webhook.from_url("Discord_WebHook_Here", adapter=RequestsWebhookAdapter())
now = datetime.now()
now_time = now.strftime("%Y-%m-%d")
client = Client()

base_info = client.get_exchange_info()
symbols = [x['symbol'] for x in base_info['symbols']]
exclude = ['UP', 'DOWN', 'BEAR', 'BULL']
non_lev = [symbol for symbol in symbols if all(excludes not in symbol for excludes in exclude)]
relevant = [symbol for symbol in non_lev if symbol.endswith('USDT')]

klines = {}
for symbol in tqdm(relevant):
    klines[symbol] = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1MINUTE, "1 day ago UTC")
#%
#% Base Variables
# webhook.send('Scheduled Sharpe calculations and MC Simulation started.')
# webhook.send('Further, USDT will be used as the base for these calculations.')

coin_api_url = 'https://rest.coinapi.io/v1/symbols?filter_symbol_id=BINANCE_SPOT_' #Pulling all symbols listed
coin_api_headers = {'X-CoinAPI-Key' : 'API_KEY_HERE'}
#% API Call (only activate when necessary)
response = requests.get(coin_api_url, headers=coin_api_headers)
coinmarket_cap_url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest?start=1&limit=200&convert=USD&sort=market_cap&sort_dir=desc'
coinmarket_header = {'X-CMC_PRO_API_KEY': 'API_KEY_HERE'}

#% Find Cryptos by market cap
coinmarket_by_cap = requests.get(coinmarket_cap_url, headers=coinmarket_header)  
jsondata = json.loads(coinmarket_by_cap.text)
df_all_symbols = json_normalize(response.json())
#%
returns, symbols = [],[]
for symbol in relevant:
    if len(klines[symbol])>0:
        cumulative_return = (pd.DataFrame(klines[symbol])[4].astype(float).pct_change() +1).prod()-1
        cumulative_return = round(cumulative_return*100,2)
        returns.append(cumulative_return)
        symbols.append(symbol)

ret_df = pd.DataFrame(returns, index = symbols, columns=['pct_ret'])

top_10 = ret_df.pct_ret.nlargest(10)
top_10_list = top_10.index.tolist()
df_all_USDT = df_all_symbols.loc[df_all_symbols['asset_id_quote'] == 'USDT']
#%
df_all_USDT = df_all_USDT.loc[df_all_USDT["asset_id_base"]!='BUSD'] #filter out BUSD
#%
# df_all_BTC = df_all_symbols.loc[df_all_symbols['asset_id_quote'] == 'BTC'].dropna(subset = ['data_orderbook_start'])
#% Defining targets
time_limit = 60
top_10_list = [x[:-4] for x in top_10_list]
    
# df_top5 = df_all_USDT[['asset_id_base','price','volume_1day_usd']].sort_values('volume_1day_usd', ascending = False).head(n=5).to_string(index=False)
# USDT_top5 = df_all_USDT[['asset_id_base','price','volume_1day_usd']].sort_values('volume_1day_usd', ascending = False).head(n=5)
output_table = df_all_USDT[df_all_USDT['asset_id_base'].isin(top_10_list)].head(n=3)
output_table = output_table[['asset_id_base', 'price']]
output_table = output_table.rename({'asset_id_base': 'Asset','price': 'Price(USDT)'}, axis='columns')

webhook.send(f"Top 10 by performance (USDT as base) today:\nTrading pair | Cumulative % change\n{top_10}\nScheduled Sharpe calculations and MC Simulation started.\nComparting top 3 by performance (USDT as base) for the last 24 hours:\n{output_table.to_string(index=False)}")
#%
lst_target_coin = output_table.Asset.tolist()
#
df_target_symbol = []
for i in lst_target_coin:
    df_target_symbol.append(df_all_USDT.loc[df_all_USDT['asset_id_base'] == i])
df_target_symbol = pd.concat(df_target_symbol)
lst_found = df_target_symbol['asset_id_base'].tolist()
if lst_target_coin == lst_found:
    print('All targeted assets found.')
else:
    webhook.send(f'\nConfirmed to be able to find the following {len(lst_found)} assets:')
    webhook.send(lst_found)
    lst_lost = list(set(lst_target_coin)-set(lst_found))
    if len(lst_lost) >0:
        webhook.send(f'\nHowever, unable to find these asset(s) {lst_lost}.')
df_target_symbol = df_target_symbol.sort_values(by=['data_start'], ascending = False)

#Defining date bounds
youngest = df_target_symbol.iloc[0].to_dict()
youngest_data_start = datetime.strptime(youngest.get("data_start"), "%Y-%m-%d")
date_diff = (now - datetime.strptime(youngest.get("data_start"), "%Y-%m-%d")).days
# webhook.send(f'Youngest asset is {youngest.get("asset_id_base")} having been listed on {youngest.get("data_start")}, or {date_diff} days back.')

#Mode choose to define the start date ourselves or go as far back as possible?
if date_diff >time_limit:
    data_start_target = now - timedelta(time_limit)
    # webhook.send(f"Youngest asset is {youngest.get('asset_id_base')}, listed on {youngest.get('data_start')}, or {date_diff} days back. To limit the number of API calls for the assets chosen to ~{math.ceil(len(lst_found)*math.ceil(date_diff/100))} calls, data collected will start at {data_start_target.strftime('%Y-%m-%d')}.")
else:
    data_start_target = youngest_data_start 
    # webhook.send(f"Youngest asset is {youngest.get('asset_id_base')}, listed on {youngest.get('data_start')}, or {date_diff} days back. Since it is only up to {math.ceil(len(lst_found)*math.ceil(date_diff/100))} call(s) in total, setting data start target to {data_start_target.strftime('%Y-%m-%d')}")
    data_start_target = youngest_data_start

#Now with data_start_target defined, can now start calling historical data
hist_response = {}
for i in range(0,len(lst_found)):
    hist_response[lst_found[i]] = ""
    hist_url = f'https://rest.coinapi.io/v1/ohlcv/BINANCE_SPOT_{lst_found[i]}_USDT/history?period_id=1DAY&time_start={datetime.strftime(data_start_target, "%Y-%m-%d")}T00:00:00&limit={time_limit}' #Pulling historical data
    hist_response[lst_found[i]] = requests.get(hist_url, headers = coin_api_headers)
    #webhook.send(f"\nCalling API for {lst_found[i]} data costed {hist_response[lst_found[i]].headers['x-ratelimit-request-cost']} calls")
    #webhook.send(f"API Call #{hist_response[lst_found[i]].headers['x-ratelimit-used']} out of {hist_response[lst_found[i]].headers['x-ratelimit-limit']}")

num_ports = 100000
    
# Generate df of each asset mainly getting day_close/day_open to determine % return, as well as Sharpe Ratio
pd.options.mode.chained_assignment = None

df_pct_change = pd.DataFrame()
df_hist_close = pd.DataFrame()
for i in range(0,len(lst_found)):
    df_hist_data = json_normalize(hist_response[lst_found[i]].json())
    df_hist_data = df_hist_data[['time_period_start','price_close']]
    df_pct_change[f"{lst_found[i]}"] = (df_hist_data['price_close'].pct_change())
    df_hist_close[f"{lst_found[i]}"] = (df_hist_data['price_close'])

# Defining some functions for calculations later
def std_dev(data):
    # Get number of observations
    n = len(data)
    # Calculate mean
    mean = sum(data) / n
    # Calculate deviations from the mean
    deviations = sum([(x - mean)**2 for x in data])
    # Calculate Variance & Standard Deviation
    variance = deviations / (n - 1)
    s = variance**(1/2)
    return s

def sharpe_ratio(data, risk_free_rate=0.0):
    # Calculate Average Daily Return
    mean_daily_return = sum(data) / len(data)
    # Calculate Standard Deviation
    s = std_dev(data)
    # Calculate Daily Sharpe Ratio
    daily_sharpe_ratio = (mean_daily_return - risk_free_rate) / s
    # Annualize Daily Sharpe Ratio
    sharpe_ratio = 365**(1/2) * daily_sharpe_ratio
    
    return sharpe_ratio
#Determining Sharpe ratios
target_asset_sharpe = {}
for i in lst_found:
    target_hist = df_pct_change[i].dropna().tolist()
    target_asset_sharpe[i] = sharpe_ratio(target_hist, risk_free_rate = 0.0)
    
# Starting Monte Carlo Distribution Testing
# Monte Carlo Distribution: Generating random weights, and then rebalancing to sum up to 1.0
weights = np.array(np.random.random(len(lst_found)))
weights = weights/np.sum(weights)

log_ret = np.log(df_hist_close/df_hist_close.shift(1))
log_ret_details = log_ret.describe().transpose()
log_ret_mean = log_ret.mean()*365
log_ret_cov = log_ret.cov()*365

exp_ret = np.sum(log_ret.mean()*weights)*365
exp_vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 365, weights)))
exp_sharpe = exp_ret/exp_vol

sim_start_time = datetime.now()
# webhook.send(f'\nRunning {num_ports:,} simulated portfolios. Simulation started at {sim_start_time.strftime("%Y.%m.%d | %H:%M:%S")}.')

# Note: Central 

all_weights = np.zeros((num_ports,len(lst_found)))
ret_arr = np.zeros(num_ports)
vol_arr = np.zeros(num_ports)
sharpe_arr = np.zeros(num_ports)

for ind in range(num_ports):
    # Create Random Weights
    weights = np.array(np.random.random(len(lst_found)))
    # Rebalance Weights
    weights = weights / np.sum(weights)    
    # Save Weights
    all_weights[ind,:] = weights
    # Expected Return
    ret_arr[ind] = np.sum((log_ret.mean() * weights) *365)
    # Expected Variance
    vol_arr[ind] = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 365, weights)))
    # Sharpe Ratio
    sharpe_arr[ind] = ret_arr[ind]/vol_arr[ind]
    # Print progress
    # progress = f'{round((ind/num_ports)*100,2)}% complete. | {ind:,}/{num_ports:,}'
    # sys.stdout.write('\r'+progress)
    # sys.stdout.flush()

# sys.stdout.flush()
# webhook.send(f'\rAll {num_ports} portfolios simulated.')
max_sharpe_ratio = sharpe_arr.max() #double check the vol_arr and arrange that for minimal vol to find min_vol
min_vol = ret_arr[np.argmin(vol_arr)]
max_sharpe = sharpe_arr.argmax()
port_dist = all_weights[max_sharpe,:]
dict_port_dist = dict(zip(lst_found,port_dist))
# for key, value in target_asset_sharpe.items():
#     webhook.send(key,': ~', round(value,2))
# for key, value in dict_port_dist.items():
#     webhook.send(key,': ~', round(value*100,2),'%')
max_sr_ret = ret_arr[max_sharpe]
max_sr_vol = vol_arr[max_sharpe]
#
output_sharpe = [{k: round(v, 2) for k, v in target_asset_sharpe.items()}]
webhook.send(f'\nThe Sharpe Ratios of the chosen coins were:\n{output_sharpe}\nThis portfolio has an overall Sharpe Ratio of {round(max_sharpe_ratio,3)}\nImplying a return of ~{round(max_sr_ret,3)} with an expected risk/volatility of ~{round(max_sr_vol,3)}')
# webhook.send(f'The data used to graph this was goes from {data_start_target} to the end of the most recent full trading day.')
# Mathematical Optimisation
def get_ret_vol_sr(weights):
    """
    Takes in weights, returns array or return,volatility, sharpe ratio
    """
    weights = np.array(weights)
    ret = np.sum(log_ret.mean() * weights) * 365
    vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 365, weights)))
    sr = ret/vol
    return np.array([ret,vol,sr])

def neg_sharpe(weights):
    return  get_ret_vol_sr(weights)[2] * -1
# Contraints
def check_sum(weights):
    '''
    Returns 0 if sum of weights is 1.0
    '''
    return np.sum(weights) - 1
# By convention of minimize function it should be a function that returns zero for conditions
cons = ({'type':'eq','fun': check_sum})
# 0-1 bounds for each weight
bounds = tuple((0,1) for x in weights)
# Initial Guess (equal distribution)
init_guess = len(lst_found) * [1./len(lst_found),]
# Sequential Least SQuares Programming (SLSQP).
optimal_variance=minimize(neg_sharpe,
                          init_guess,
                          method = 'SLSQP',
                          bounds = bounds,
                          constraints = cons)

optimal_variance_weights=optimal_variance['x'].round(4)
list(zip(lst_found,list(optimal_variance_weights)))

opt_vol = get_ret_vol_sr(optimal_variance.x)

frontier_ret = np.linspace(min_vol ,max_sr_ret,100)

def minimize_volatility(weights):
    return  get_ret_vol_sr(weights)[1]
 
frontier_vol = []

for possible_return in frontier_ret:
    cons = ({'type':'eq', 'fun':check_sum},
            {'type':'eq', 'fun': lambda w: get_ret_vol_sr(w)[0] - possible_return})
    
    result = minimize(minimize_volatility,init_guess,method='SLSQP', bounds=bounds, constraints=cons)
    frontier_vol.append(result['fun'])

sim_end_time = datetime.now()
sim_time_taken = sim_end_time - sim_start_time
# webhook.send(f'Simulations completed in ~{str(sim_time_taken)}')
#%% Plotting
# webhook.send('Generating scatter plot/Markowitz Curve.')
graph_start_time = datetime.now()
fig = plt.figure(facecolor='black', figsize=(16,16))
ax = plt.subplot2grid((4,4),(0,0), rowspan =4, colspan = 4, facecolor = 'k')
plt.style.use('dark_background')
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='inferno')
plt.colorbar(label='Sharpe Ratio')
ax.scatter(frontier_vol,frontier_ret, color = 'azure', s=30, label='Efficient Frontier')
max_sharpe = plt.scatter(max_sr_vol,max_sr_ret, c= 'red', marker='X')
ax.grid(True, color = 'w', linestyle=':')
plt.xlabel('Volatility/Risk (Standard Deviation)', color = 'w')
plt.ylabel('Expected Return', color = 'w')
plt.legend()
plt.title(f'Markowitz Efficient Frontier for the top 3 cryptocurrencies by cumulative returns over the last day on Binance from {data_start_target.strftime("%Y.%m.%d")} to {now_time}\nPortfolio with the maximised Sharpe Ratio of ~{round(max_sharpe_ratio,2)} (red dot) - [Expected Return ~{round(max_sr_ret,2)} | Volatility/Risk ~{round(max_sr_vol,2)}]\n*DO NOT USE AS FINANCIAL ADVICE*')
col_labels=['Portfolio Distribution']
row_labels= [str(str(k)+' ('+str(round(v,2))+')') for k, v in target_asset_sharpe.items()]
table_port_dist = []
for i in port_dist:
    table_port_dist.append([str(round(i*100,2))+'%'])
# the rectangle is where I want to place the table
the_table = plt.table(cellText=table_port_dist,
                      cellColours = ['k','k','k'],
                      colWidths = [0.13],
                      cellLoc = 'center',
                      rowLoc = 'right',                      
                      rowLabels=row_labels,
                      rowColours= ['k','k','k'],
                      colLabels=col_labels,
                      colColours = 'k',
                      loc='lower right')
plt.savefig(f'top_gainers_{now_time}_result.jpg')

outputfile = f'top_gainers_{now_time}_result.jpg'

found_string = ''
for i in lst_found:
    found_string = found_string+' #'+i

webhook.send('Uploaded to instagram, now cleaning up', file = File(outputfile))

#%%
insta_username = "Username_Here"
insta_password = "Password_Here"
upload_target=outputfile
post_caption = f"Update as of {now_time}.\nSharpe Ratio of these cryptocurrencies calculated based on daily returns.\nPortfolio distributions referred to within the output is the theoretical 'optimised/efficient' portfolio according to Modern Portfolio Theory found through the application of Monte Carlo Method.\nAs a base, the start date for analysis defined by youngest asset ({youngest.get('asset_id_base')}) if less than {time_limit} days.\nThe selected cryptos and their price as of this analysis:\n{output_table.to_string(index=False)}\n#notfinancialadvice #crypto #cryptocurrencies #MarkowitzFrontier #ModernPortfolioTheory #EfficientFrontier #SharpeRatio #selftaught #automation {found_string}"
bot = Bot()
bot.login(username=insta_username, password=insta_password)
bot.upload_photo(upload_target, caption= post_caption)
# with insta_client(insta_username, insta_password) as cli:
#     cli.upload(upload_target, post_caption)
#%
for f in os.listdir('.'):
    if f.endswith('.REMOVE_ME'):
        os.remove(f)

exit()
