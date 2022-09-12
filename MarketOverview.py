# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 12:42:06 2022

@author: User
"""

import ta, os, matplotlib, sys
import pandas as pd
from binance.client import Client
from datetime import datetime
from instagrapi import Client as instabot
import matplotlib.pyplot as plt
from discord import Webhook, RequestsWebhookAdapter, File

webhook = Webhook.from_url("WEBHOOK_HERE", adapter=RequestsWebhookAdapter())
now = datetime.now()
now_time = now.strftime("%Y-%m-%d")

client = Client()

trade_pair = 'BTCUSDT'

def neg_or_pos(number):
    if number >0:
        return "suggests Bull"
    else:
        return "suggests Bear"

def gethistdata_week(symbol, interval, lookback):
    frame = pd.DataFrame(client.get_historical_klines(symbol,
                                                      interval,
                                                      lookback + 'week ago UTC'))
    frame = frame.iloc[:,:6]
    frame.columns = ['Time','Open', 'High', 'Low', 'Close', 'Volume']
    frame = frame.set_index('Time')
    frame.index = pd.to_datetime(frame.index, unit = 'ms')
    frame = frame.astype(float)
    return frame

def gethistdata_day(symbol, interval, lookback):
    frame = pd.DataFrame(client.get_historical_klines(symbol,
                                                      interval,
                                                      lookback + 'day ago UTC'))
    frame = frame.iloc[:,:6]
    frame.columns = ['Time','Open', 'High', 'Low', 'Close', 'Volume']
    frame = frame.set_index('Time')
    frame.index = pd.to_datetime(frame.index, unit = 'ms')
    frame = frame.astype(float)
    return frame

def applytechnicals_week(df):
    df['KLine'] = ta.momentum.stoch(df.High, df.Low, df.Close, window = 14, smooth_window=3) #Calculate K Line
    df['DLine'] = df['KLine'].rolling(3).mean() #D Line, the SMA over the K Line
    df['macd']=ta.trend.macd_diff(df.Close)
    indicator_ichimoku = ta.trend.IchimokuIndicator(df.High, df.Low, window1 = 9, window2 = 26, window3 = 52, visual = True, fillna = False)
    df['ichi_senkouA']= indicator_ichimoku.ichimoku_a()
    df['ichi_senkouB']= indicator_ichimoku.ichimoku_b()
    df['ichi_conversion']= indicator_ichimoku.ichimoku_conversion_line() #Tenkan
    df['ichi_base']= indicator_ichimoku.ichimoku_base_line() #Kijun
    df['EMA7'] = ta.trend.EMAIndicator(df.Close, window = 7, fillna=True).ema_indicator()
    df['EMA25'] = ta.trend.EMAIndicator(df.Close, window = 25, fillna=True).ema_indicator()
    df['SMA2Y'] = ta.trend.SMAIndicator(df.Close, window = 104, fillna=True).sma_indicator()
    df['SMA2Yx5'] = 5*(ta.trend.SMAIndicator(df.Close, window = 104, fillna=True).sma_indicator())
    df['4weekheatmap'] = round(df['SMA2Y'].pct_change(periods = 4, fill_method = 'bfill', limit = None)*100, 2)
    df.dropna(inplace=True)

def applytechnicals_day(df):
    df['KLine'] = ta.momentum.stoch(df.High, df.Low, df.Close, window = 14, smooth_window=3) #Calculate K Line
    df['DLine'] = df['KLine'].rolling(3).mean() #D Line, the SMA over the K Line
    df['macd']=ta.trend.macd_diff(df.Close)
    indicator_ichimoku = ta.trend.IchimokuIndicator(df.High, df.Low, window1 = 9, window2 = 26, window3 = 52, visual = True, fillna = False)
    df['ichi_senkouA']= indicator_ichimoku.ichimoku_a()
    df['ichi_senkouB']= indicator_ichimoku.ichimoku_b()
    df['ichi_conversion']= indicator_ichimoku.ichimoku_conversion_line() #Tenkan
    df['ichi_base']= indicator_ichimoku.ichimoku_base_line() #Kijun
    df['EMA7'] = ta.trend.EMAIndicator(df.Close, window = 7, fillna=True).ema_indicator()
    df['EMA25'] = ta.trend.EMAIndicator(df.Close, window = 25, fillna=True).ema_indicator()
    df['SMA2Y'] = ta.trend.SMAIndicator(df.Close, window = 730, fillna=True).sma_indicator()
    df['SMA2Yx5'] = 5*(ta.trend.SMAIndicator(df.Close, window = 730, fillna=True).sma_indicator())
    df['4weekheatmap'] = round(df['SMA2Y'].pct_change(periods = 28, fill_method = 'bfill', limit = None)*100, 2)
    df.dropna(inplace=True)

time_unit = '1w'
time_period = '129'
df_week = gethistdata_week(trade_pair, time_unit, time_period)
applytechnicals_week(df_week)
plt.figure(facecolor='black', figsize=(16,16))
plt.style.use('dark_background')
plt.gca().yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
plt.title(f'{trade_pair} weekly lookback from {df_week.index[0].strftime("%Y-%m-%d")} to {df_week.index[-1].strftime("%Y-%m-%d")}\nColor bar showing 4w % change of 2Y SMA as a heatmap', color = 'w')
plt.fill_between(df_week.index, df_week.ichi_senkouA, df_week.ichi_senkouB, where =(df_week.ichi_senkouA>df_week.ichi_senkouB), color = 'g', alpha = 0.3, interpolate = True)
plt.fill_between(df_week.index, df_week.ichi_senkouB, df_week.ichi_senkouA, where =(df_week.ichi_senkouB>df_week.ichi_senkouA), color = 'r', alpha = 0.3, interpolate = True)
plt.plot(df_week.Close, color = 'w', alpha=1, label='Price')
plt.plot(df_week.ichi_base, color = 'w', alpha=0.5, label='Ichimoku - Base')
plt.plot(df_week.ichi_conversion, color = 'dodgerblue', alpha=0.5, label='Ichimoku - Conversion')
plt.plot(df_week.ichi_senkouA, color = 'g', alpha=0.8)
plt.plot(df_week.ichi_senkouB, color = 'r', alpha=0.8)
plt.plot(df_week.EMA7, color = 'yellow', label='EMA (7)')
plt.plot(df_week.EMA25, color = 'purple', label='EMA (25)')
plt.plot(df_week.SMA2Y, color = 'gainsboro', label='2 Year SMA')
plt.plot(df_week.SMA2Yx5, color = 'cyan', label='2 Year SMA x5')
plt.scatter(df_week.index, df_week.Close, s=30, c=df_week['4weekheatmap'], cmap='jet')
plt.colorbar()
plt.legend(loc = 'best')
plt.savefig(f'{trade_pair}_{now_time}_weekly_result.jpg')
#plt.show()
weekly_outputfile = f'{trade_pair}_{now_time}_weekly_result.jpg'
webhook.send(f'Weekly period:\n{trade_pair} Price at Close for last full week: ${df_week.Close[-2]:,}\nPrice in relation to 2Y-MA=>2Y-MAx5: {round(((df_week.Close[-2]-df_week.SMA2Y[-2])/(df_week.SMA2Yx5[-2]-df_week.SMA2Y[-2]))*100,1)}%\nWeekly EMA indicator (7 periods vs 25 periods) {neg_or_pos(df_week.EMA7[-2]-df_week.EMA25[-2])} [Δ of {round(((df_week.EMA7[-2]-df_week.EMA25[-2])/df_week.EMA25[-2])*100,2)}% between the two EMA lines]', file = File(weekly_outputfile))

time_unit = '1d'
time_period = '599'
df_day = gethistdata_day(trade_pair, time_unit, time_period)
applytechnicals_day(df_day)
plt.figure(facecolor='black', figsize=(16,16))
plt.style.use('dark_background')
plt.gca().yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
plt.title(f'{trade_pair} daily lookback from {df_day.index[0].strftime("%Y-%m-%d")} to {df_day.index[-1].strftime("%Y-%m-%d")}\nColor bar showing 4w % change of 2Y SMA as a heatmap', color = 'w')
plt.fill_between(df_day.index, df_day.ichi_senkouA, df_day.ichi_senkouB, where =(df_day.ichi_senkouA>df_day.ichi_senkouB), color = 'g', alpha = 0.3, interpolate = True)
plt.fill_between(df_day.index, df_day.ichi_senkouB, df_day.ichi_senkouA, where =(df_day.ichi_senkouB>df_day.ichi_senkouA), color = 'r', alpha = 0.3, interpolate = True)
plt.plot(df_day.Close, color = 'w', alpha=1, label='Price')
plt.plot(df_day.ichi_base, color = 'w', alpha=0.5, label='Ichimoku - Base')
plt.plot(df_day.ichi_conversion, color = 'dodgerblue', alpha=0.5, label='Ichimoku - Conversion')
plt.plot(df_day.ichi_senkouA, color = 'g', alpha=0.8)
plt.plot(df_day.ichi_senkouB, color = 'r', alpha=0.8)
plt.plot(df_day.EMA7, color = 'yellow', label='EMA (7)')
plt.plot(df_day.EMA25, color = 'purple', label='EMA (25)')
plt.plot(df_day.SMA2Y, color = 'gainsboro', label='2 Year SMA')
plt.plot(df_day.SMA2Yx5, color = 'cyan', label='2 Year SMA x5')
plt.scatter(df_day.index, df_day.Close, s=30, c=df_day['4weekheatmap'], cmap='jet')
plt.colorbar()
plt.legend(loc = 'best')
plt.savefig(f'{trade_pair}_{now_time}_daily_result.jpg')
#plt.show()
daily_outputfile = f'{trade_pair}_{now_time}_daily_result.jpg'
webhook.send(f'Daily period:\n{trade_pair} Price currently: ${df_day.Close[-1]:,}\nPrice in relation to 2Y-MA=>2Y-MAx5: {round(((df_day.Close[-1]-df_day.SMA2Y[-1])/(df_day.SMA2Yx5[-1]-df_day.SMA2Y[-1]))*100,1)}%\nDaily EMA indicator (7 periods vs 25 periods) {neg_or_pos(df_day.EMA7[-1]-df_day.EMA25[-1])} [Δ of {round(((df_day.EMA7[-1]-df_day.EMA25[-1])/df_day.EMA25[-1])*100,2)}% between the EMA lines of the two periods]', file = File(daily_outputfile))

trade_pair = 'ETHUSDT'

time_unit = '1w'
time_period = '129'
df_week_eth = gethistdata_week(trade_pair, time_unit, time_period)
applytechnicals_week(df_week_eth)
plt.figure(facecolor='black', figsize=(16,16))
plt.style.use('dark_background')
plt.gca().yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
plt.title(f'{trade_pair} weekly lookback from {df_week_eth.index[0].strftime("%Y-%m-%d")} to {df_week_eth.index[-1].strftime("%Y-%m-%d")}\nColor bar showing 4w % change of 2Y SMA as a heatmap', color = 'w')
plt.fill_between(df_week_eth.index, df_week_eth.ichi_senkouA, df_week_eth.ichi_senkouB, where =(df_week_eth.ichi_senkouA>df_week_eth.ichi_senkouB), color = 'g', alpha = 0.3, interpolate = True)
plt.fill_between(df_week_eth.index, df_week_eth.ichi_senkouB, df_week_eth.ichi_senkouA, where =(df_week_eth.ichi_senkouB>df_week_eth.ichi_senkouA), color = 'r', alpha = 0.3, interpolate = True)
plt.plot(df_week_eth.Close, color = 'w', alpha=1, label='Price')
plt.plot(df_week_eth.ichi_base, color = 'w', alpha=0.5, label='Ichimoku - Base')
plt.plot(df_week_eth.ichi_conversion, color = 'dodgerblue', alpha=0.5, label='Ichimoku - Conversion')
plt.plot(df_week_eth.ichi_senkouA, color = 'g', alpha=0.8)
plt.plot(df_week_eth.ichi_senkouB, color = 'r', alpha=0.8)
plt.plot(df_week_eth.EMA7, color = 'yellow', label='EMA (7)')
plt.plot(df_week_eth.EMA25, color = 'purple', label='EMA (25)')
plt.plot(df_week_eth.SMA2Y, color = 'gainsboro', label='2 Year SMA')
plt.plot(df_week_eth.SMA2Yx5, color = 'cyan', label='2 Year SMA x5')
plt.scatter(df_week_eth.index, df_week_eth.Close, s=30, c=df_week_eth['4weekheatmap'], cmap='jet')
plt.colorbar()
plt.legend(loc = 'best')
plt.savefig(f'{trade_pair}_{now_time}_weekly_result.jpg')
#plt.show()
weekly_outputfile = f'{trade_pair}_{now_time}_weekly_result.jpg'
webhook.send(f'Weekly period:\n{trade_pair} Price at Close for last full week: ${df_week_eth.Close[-2]:,}\nPrice in relation to 2Y-MA=>2Y-MAx5: {round(((df_week_eth.Close[-2]-df_week_eth.SMA2Y[-2])/(df_week_eth.SMA2Yx5[-2]-df_week_eth.SMA2Y[-2]))*100,1)}%\nWeekly EMA indicator (7 periods vs 25 periods) {neg_or_pos(df_week_eth.EMA7[-2]-df_week_eth.EMA25[-2])} [Δ of {round(((df_week_eth.EMA7[-2]-df_week_eth.EMA25[-2])/df_week_eth.EMA25[-2])*100,2)}% between the two EMA lines]', file = File(weekly_outputfile))

time_unit = '1d'
time_period = '599'
df_day_eth = gethistdata_day(trade_pair, time_unit, time_period)
applytechnicals_day(df_day_eth)
plt.figure(facecolor='black', figsize=(16,16))
plt.style.use('dark_background')
plt.gca().yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
plt.title(f'{trade_pair} daily lookback from {df_day_eth.index[0].strftime("%Y-%m-%d")} to {df_day_eth.index[-1].strftime("%Y-%m-%d")}\nColor bar showing 4w % change of 2Y SMA as a heatmap', color = 'w')
plt.fill_between(df_day_eth.index, df_day_eth.ichi_senkouA, df_day_eth.ichi_senkouB, where =(df_day_eth.ichi_senkouA>df_day_eth.ichi_senkouB), color = 'g', alpha = 0.3, interpolate = True)
plt.fill_between(df_day_eth.index, df_day_eth.ichi_senkouB, df_day_eth.ichi_senkouA, where =(df_day_eth.ichi_senkouB>df_day_eth.ichi_senkouA), color = 'r', alpha = 0.3, interpolate = True)
plt.plot(df_day_eth.Close, color = 'w', alpha=1, label='Price')
plt.plot(df_day_eth.ichi_base, color = 'w', alpha=0.5, label='Ichimoku - Base')
plt.plot(df_day_eth.ichi_conversion, color = 'dodgerblue', alpha=0.5, label='Ichimoku - Conversion')
plt.plot(df_day_eth.ichi_senkouA, color = 'g', alpha=0.8)
plt.plot(df_day_eth.ichi_senkouB, color = 'r', alpha=0.8)
plt.plot(df_day_eth.EMA7, color = 'yellow', label='EMA (7)')
plt.plot(df_day_eth.EMA25, color = 'purple', label='EMA (25)')
plt.plot(df_day_eth.SMA2Y, color = 'gainsboro', label='2 Year SMA')
plt.plot(df_day_eth.SMA2Yx5, color = 'cyan', label='2 Year SMA x5')
plt.scatter(df_day_eth.index, df_day_eth.Close, s=30, c=df_day_eth['4weekheatmap'], cmap='jet')
plt.colorbar()
plt.legend(loc = 'best')
plt.savefig(f'{trade_pair}_{now_time}_daily_result.jpg')
#plt.show()
daily_outputfile = f'{trade_pair}_{now_time}_daily_result.jpg'
webhook.send(f'Daily period:\n{trade_pair} Price currently: ${df_day_eth.Close[-1]:,}\nPrice in relation to 2Y-MA=>2Y-MAx5: {round(((df_day_eth.Close[-1]-df_day_eth.SMA2Y[-1])/(df_day_eth.SMA2Yx5[-1]-df_day_eth.SMA2Y[-1]))*100,1)}%\nDaily EMA indicator (7 periods vs 25 periods) {neg_or_pos(df_day_eth.EMA7[-1]-df_day_eth.EMA25[-1])} [Δ of {round(((df_day_eth.EMA7[-1]-df_day_eth.EMA25[-1])/df_day_eth.EMA25[-1])*100,2)}% between the EMA lines of the two periods]', file = File(daily_outputfile))
#%%

for f in os.listdir('.'):
    if f.endswith('.jpg'):
        os.remove(f)

sys.exit()
