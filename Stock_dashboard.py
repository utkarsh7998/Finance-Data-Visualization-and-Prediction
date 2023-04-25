from stocknews import StockNews
from alpha_vantage.fundamentaldata import FundamentalData
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
import ipywidgets as widgets
import datetime
from datetime import date, timedelta, datetime
import time
from prophet import Prophet
from prophet.plot import plot_plotly
import seaborn as sns
import requests
#ML Model
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential,load_model
st.set_page_config(
    page_title="Stock Dashboard",
    page_icon="chart_with_upwards_trend",  # "🧊",
    layout="wide",
    menu_items={
        'About': "This is part of CS661A course project , IIT Kanpur. Submitted by Group-23. Stock DashBoard is a web application that allows you to visualize stock data and perform analysis using various Fundamental and Technical indicators. Team Members: Abhishek Sahu(21111002), Archit Gupta, Susmitha, Sai Kiran and Utkarsh"
    }
)
# # Section 1
# button = st.button('Button')
# button_placeholder = st.empty()
# button_placeholder.write(f'button = {button}')
# time.sleep(2)
# button = False
# button_placeholder.write(f'button = {button}')

# # Section 2
# time_placeholder = st.empty()

# while True:
#     timenow = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     time_placeholder.write(timenow)
#     time.sleep(1)
st.title('Stock Dashboard')
ticker = st.sidebar.text_input('Stock Ticker', 'GOOG')
utkarsh_ticker = ticker
today = date.today()
default_date_yesterday = today - timedelta(days=2*365)
start_date = st.sidebar.date_input('Start Date', default_date_yesterday)
end_date = st.sidebar.date_input('End Date', today)
# if end_date not equal to today then add 1 day to end date
d1 = end_date.strftime('%d-%m-%Y')
d2 = today.strftime('%d-%m-%Y')
if d1 != d2:
    end_date_new = end_date + timedelta(days=1)
else:
    end_date_new = end_date
data = yf.download(ticker, start=start_date, end=end_date_new)
# st.write(data)
# find the days difference between start and end date
num_days = (end_date - start_date).days
st.sidebar.write('Number of days:', num_days)

# Select the required time period
values = ['1D', '1W', '1Y', "3Y", "5Y", "Max"]
default_ix = values.index("1Y")
sel_metric = st.sidebar.selectbox(
    'Select the required time delta',
    values, index=default_ix)

if sel_metric == '1D':
    start_date = end_date - timedelta(days=1)
elif sel_metric == '1W':
    start_date = end_date - timedelta(days=6)
elif sel_metric == '1Y':
    start_date = end_date - timedelta(days=364)
elif sel_metric == '3Y':
    start_date = end_date - timedelta(days=3*365-1)
elif sel_metric == '5Y':
    start_date = end_date - timedelta(days=5*365-1)
elif sel_metric == 'Max':
    start_date = start_date


col1, col2 = st.sidebar.columns(2)
col1.write('Delta Start Date:')
col2.write('Delta End Date:')

# col1, col2 = st.sidebar.columns(2)
col1.write(start_date.strftime('%d-%m-%Y'))
col2.write(end_date.strftime('%d-%m-%Y'))

# col1, col2 = st.sidebar.columns(2)

start_date = start_date.strftime('%Y-%m-%d %H:%M:%S')
end_date = end_date.strftime('%Y-%m-%d %H:%M:%S')
if start_date in data.index:
    col1.markdown(
        'Delta Start Data Found: <span style="color:green">True</span>', unsafe_allow_html=True)
else:
    col1.markdown(
        'Delta Start Data Found: <span style="color:red">False</span>', unsafe_allow_html=True)

if end_date in data.index:
    col2.markdown(
        'Delta End Data Found: <span style="color:green">True</span>', unsafe_allow_html=True)
else:
    col2.markdown(
        'Delta End Data Found: <span style="color:red">False</span>', unsafe_allow_html=True)


# check if start_date is not in data.index
if start_date not in data.index:
    start_date = data.index[data.index.get_loc(start_date, method='nearest')]
    st.sidebar.write('Nearest Delta Start Date:', start_date)
# else:
#     st.sidebar.write('Start Date Data:', start_date)
    # st.sidebar.write('Nearest Start Date Data:', data.loc[start_date])

# st.sidebar.write('End Date:', end_date)

if end_date not in data.index:
    end_date = data.index[data.index.get_loc(end_date, method='nearest')]
    st.sidebar.write('Nearest Delta End Date:', end_date)
    # st.sidebar.write('Nearest End Date Data:', data.loc[end_date])
# else:
#     st.sidebar.write('End Date Data:', end_date)

col1, col2 = st.sidebar.columns(2)
col1.write('Delta Start Data:')
col2.write('Delta End Data:')
col1.write(data.loc[start_date])
col2.write(data.loc[end_date])


col1, col2, col3, col4 = st.columns(4, gap="small")
high_diff = round(data.loc[end_date]["High"] - data.loc[start_date]["High"], 2)
high_diff_percentage = abs(
    round((high_diff/data.loc[start_date]["High"])*100, 2))
col1.metric("High (" + sel_metric + " )", round(data.loc[end_date]["High"], 2), str(
    high_diff)+" ("+str(high_diff_percentage)+"%)")

low_diff = round(data.loc[end_date]["Low"] - data.loc[start_date]["Low"], 2)
low_diff_percentage = abs(round((low_diff/data.loc[start_date]["Low"])*100, 2))
col2.metric("Low (" + sel_metric + " )", round(data.loc[end_date]["Low"], 2), str(
    low_diff)+" ("+str(low_diff_percentage)+"%)")

open_diff = round(data.loc[end_date]["Open"] -
                  data.loc[start_date]["Open"], 2)
open_diff_percentage = abs(
    round((open_diff/data.loc[start_date]["Open"])*100, 2))
col3.metric("Open (" + sel_metric + " )", round(data.loc[end_date]["Open"], 2), str(
    open_diff)+" ("+str(open_diff_percentage)+"%)")

close_diff = round(data.loc[end_date]["Close"] -
                   data.loc[start_date]["Close"], 2)
close_diff_percentage = abs(
    round((close_diff/data.loc[start_date]["Close"])*100, 2))
col4.metric("Close (" + sel_metric + " )", round(data.loc[end_date]["Close"], 2), str(
    close_diff)+" ("+str(close_diff_percentage)+"%)")


# col4.metric("Open","8","3%")

## Adding Bubble Chart on Top Here
df = pd.read_csv('BubbleData.csv')
import plotly.express as px
fig = px.scatter(df, x="Close", y="Gain", 
           animation_frame="Time",
            animation_group="Ticker",
           size="Volume", 
           color="Company", 
           hover_name="Label",
        labels={
                    "Close":"Closing Value", 
                    "Gain": "% Gain"
                 },
             size_max=100, 
             range_x=[np.min(df['Close'])-5,np.max(df['Close'])+5], range_y=[np.min(df['Gain'])-5,np.max(df['Gain'])+5]
             )
fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 600
fig.update_layout(
    title={
        'text': "Animated Bubble Chart of Top Companies",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
st.plotly_chart(fig, use_container_width=True)

# st.write(data)
options = st.multiselect(
    'Select the chart type',
    ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'],
    ['Close'])


if options == [] or options == ['Close']:
    fig1 = px.line(data, x=data.index, y=data['Close'], title=ticker, labels={
                   "x": "Date", "y": "Price"})
else:
    fig1 = px.line(data, x=data.index, y=options, title=ticker)
    fig1.update_yaxes(title=dict(text='Stock Price'))


fig2 = go.Figure(
    data=go.Ohlc(
        x=data.index,
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
    )
)
fig2.update(layout_xaxis_rangeslider_visible=False)
fig2.update_layout(
    title='Open-High-Low-Close Chart',
    xaxis_title="Date",
    yaxis_title="Price")
fig_col1, fig_col2 = st.columns(2, gap='small')
fig_col1.plotly_chart(fig1, use_container_width=True)
fig_col2.plotly_chart(fig2, use_container_width=True)

fig = go.Figure()
fig.add_trace(go.Candlestick(x=data.index,
                             open=data["Open"],
                             high=data["High"],
                             low=data["Low"],
                             close=data["Close"],
                             name=ticker))
fig.update_layout(
    title='Candle Stick Chart',
    xaxis_title="Date",
    yaxis_title="Price")
st.plotly_chart(fig, use_container_width=True)



col1, col2 = st.columns(2, gap='small')
# Calculate the daily returns for the stock
data["Daily_Return"] = (data["Close"] - data["Close"].shift(1))

# Create a box plot of the daily returns using Plotly
fig = go.Figure()
fig.add_trace(go.Box(y=data["Daily_Return"].dropna(),
                     name="Daily Returns"))

# Update the layout of the plot
fig.update_layout(title="Daily Returns",
                  yaxis_title="Return")
col1.plotly_chart(fig, use_container_width=True)

# Create a box plot of the OHLC data using Plotly
traces = []
for column in ["Open", "Close", "High", "Low"]:
    trace = go.Box(
        y=data[column],
        name=column,
        boxpoints="outliers",
        jitter=0.3,
        whiskerwidth=0.2,
        marker=dict(size=2),
        line=dict(width=1),
    )
    traces.append(trace)

# Create the layout for the plot
layout = go.Layout(
    title="Box Plot of OHLC",
    yaxis=dict(title="Price"),
    boxmode="group",
)

# Create the figure and plot the data
fig = go.Figure(data=traces, layout=layout)
col2.plotly_chart(fig, use_container_width=True)
# penguins = sns.load_dataset("penguins")

# st.title("Hello")
# fig = sns.pairplot(penguins, hue="species")
# st.pyplot(fig)
# st.write(data)

col1, col2 = st.columns(2)
values = ['None','OHLC', 'Candle Stick Chart', 'Box Plot',
          'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
default_ix = values.index('None')
sel_metric = col1.selectbox(
    'See Explanation for:',
    values, index=default_ix)
explanations = [
                """""",
                """An OHLC chart is a type of bar chart that shows open, high, low, and closing prices for each period. OHLC charts are useful since they show the four major data points over a period, with the closing price being considered the most important by many traders. The chart type is useful because it can show increasing or decreasing momentum. When the open and close are far apart it shows strong momentum, and when the open and close are close together it shows indecision or weak momentum. The high and low show the full price range of the period, useful in assessing volatility.""" , 
                """A Candlestick Chart is a type of price chart used in technical analysis that displays the high, low, open, and closing prices of a security for a specific period. The wide part of the candlestick is called the "real body" and tells investors whether the closing price was higher or lower than the opening price (black/red if the stock closed lower, white/green if the stock closed higher).""", 
                """Box Plot (or Box Chart) is a convenient way of graphically depicting groups of numerical data through their quartiles. It provides a visual representation of statistical data based on the minimum, first quartile, median, third quartile, and maximum. Outliers can be plotted on Box Plots as individual points.""", 
                """It is the price at which the financial security opens in the market when trading begins. It may or may not be different from the previous day's closing price. The security may open at a higher price than the closing price due to excess demand of the security.""",
                """Today's high refers to a security's intraday highest trading price. It is represented by the highest point on a day's stock chart. Today's high provides information to traders and investors on a stock's price, what news is driving the price that day, what might be a good entry and exit point into and out of the stock, and what the future outlook of the stock's price might be.""", 
                """Today's low is a security's intraday low trading price. Today's low is the lowest price at which a stock trades over the course of a trading day.""", 
                """The close is a reference to the end of a trading session in the financial markets when the markets close for the day.""", 
                """The adjusted closing price amends a stock's closing price to reflect that stock's value after accounting for any corporate actions.
                    The closing price is the raw price, which is just the cash value of the last transacted price before the market closes.
                    The adjusted closing price factors in corporate actions, such as stock splits, dividends, and rights offerings.
                    The adjusted closing price can obscure the impact of key nominal prices and stock splits on prices in the short term.""", 
                """Volume is the number of shares of a security traded during a given period of time.
                    Generally securities with more daily volume are more liquid than those without, since they are more "active".
                    Volume is an important indicator in technical analysis because it is used to measure the relative significance of a market move.
                    The higher the volume during a price move, the more significant the move and the lower the volume during a price move, the less significant the move."""
                ]
explanation = explanations[values.index(sel_metric)]
col1.write(explanation)
# expander = col1.expander("Explain "+sel_metric)
# expander.write(
#     explanation
# )

st.divider()


# Show 4 Most important indicators
st.header('Top 4 Important Indicators')
fig_col1, fig_col2 = st.columns(2, gap='small')
# fig_col1.plotly_chart(fig1,use_container_width=True)
# fig_col2.plotly_chart(fig2,use_container_width=True)
# 1. BBANDS

method = "bbands"
indicator = pd.DataFrame((getattr(ta, method)(
    low=data['Low'], close=data['Close'], high=data['High'], open=data['Open'], volume=data['Volume'])))
indicator['Close'] = data['Close']
figW_ind_new = px.line(indicator, title="Bollinger Bands")
fig_col1.plotly_chart(figW_ind_new, use_container_width=True)
# st.write(indicator) #Ask option to show the dataframe and description about the indicator
show_option1, show_option2 = fig_col1.columns(2)
expander = show_option1.expander("Explain BBands")
expander.write(
    """Bollinger Bands are a volatility indicator that consists of a set of three lines plotted on a price chart. 
    The middle line is a moving average, typically set at 20 periods, while the upper and lower bands are two 
    standard deviations away from the moving average. 
    The bands widen or narrow as volatility increases or decreases, respectively."""
)
agree = show_option2.checkbox('Show BBands Data')
if agree:
    fig_col1.write(indicator)

# method = "bbands"
# indicator = pd.DataFrame((getattr(ta, method)(
#     low=data['Low'], close=data['Close'], high=data['High'], open=data['Open'], volume=data['Volume'])))
# indicator['Close'] = data['Close']
# figW_ind_new = px.line(indicator, title="Bollinger Bands")
# fig_col1.plotly_chart(figW_ind_new, use_container_width=True)
# 2. RSI
method = "rsi"
indicator = pd.DataFrame((getattr(ta, method)(
    low=data['Low'], close=data['Close'], high=data['High'], open=data['Open'], volume=data['Volume'])))
indicator['Close'] = data['Close']
figW_ind_new = px.line(indicator, title="Relative Strength Index")
fig_col2.plotly_chart(figW_ind_new, use_container_width=True)
# st.write(indicator) #Ask option to show the dataframe and description about the indicator
show_option1, show_option2 = fig_col2.columns(2)
expander = show_option1.expander("Explain RSI")
expander.write(
    """RSI is a momentum oscillator that measures the strength of price movements. 
    It is calculated by comparing the average gains and losses of a security over a given time period, usually 14 days. 
    The RSI is plotted on a scale of 0 to 100, with overbought and oversold levels set at 70 and 30, respectively."""
)
agree = show_option2.checkbox('Show RSI Data')
if agree:
    fig_col2.write(indicator)
# fig_col1, fig_col2 = st.columns(2, gap='small')
# 3. MACD
method = "macd"
indicator = pd.DataFrame((getattr(ta, method)(
    low=data['Low'], close=data['Close'], high=data['High'], open=data['Open'], volume=data['Volume'])))
indicator['Close'] = data['Close']
figW_ind_new = px.line(
    indicator, title="Moving Average Convergence Divergence (MACD)")
fig_col1.plotly_chart(figW_ind_new, use_container_width=True)
# st.write(indicator) #Ask option to show the dataframe and description about the indicator
show_option1, show_option2 = fig_col1.columns(2)
expander = show_option1.expander("Explain MACD")
expander.write(
    """MACD is a trend-following momentum indicator that consists of two moving averages and a histogram. 
    The MACD line is the difference between a 12-period and a 26-period exponential moving average (EMA), while the signal line is a 9-period EMA of the MACD line. 
    The histogram represents the difference between the MACD line and the signal line. """
)
agree = show_option2.checkbox('Show MACD Data')
if agree:
    fig_col1.write(indicator)
# 4. PSL
method = "psl"
indicator = pd.DataFrame((getattr(ta, method)(
    low=data['Low'], close=data['Close'], high=data['High'], open=data['Open'], volume=data['Volume'])))
indicator['Close'] = data['Close']
figW_ind_new = px.line(indicator, title="Psychological Line")
fig_col2.plotly_chart(figW_ind_new, use_container_width=True)
# st.write(indicator) #Ask option to show the dataframe and description about the indicator
show_option1, show_option2 = fig_col2.columns(2)
expander = show_option1.expander("Explain PSL")
expander.write(
    """Psychological Line is a technical indicator that measures the sentiment of traders and investors by tracking the price level of a security. 
    It is based on the idea that certain price levels, such as round numbers or key support and resistance levels, 
    can have a psychological impact on market participants."""
)
agree = show_option2.checkbox('Show PSL Data')
if agree:
    fig_col2.write(indicator)
st.divider()
pricing_data, fundamental_data, news, tech_indicator, stocks_comparision, stock_prediction, portfolio_optimization = st.tabs(
    ['Pricing Data', 'Fundamental Analysis', 'Top 10 News', 'Technical Analysis', 'Compare Stocks', 'Predict Your Stock', 'Portfolio Optimization'])

with pricing_data:
    try:
        st.header('Price Movements')
        data2 = data
        data2['% Change'] = data['Adj Close']/data['Adj Close'].shift(1) - 1
        data2.dropna(inplace=True)
        st.write(data2)
        annual_return = data2['% Change'].mean()*252*100
        st.write('Annual Return is ', round(annual_return, 2), '%')
        stdev = np.std(data2['% Change'])*np.sqrt(252)
        st.write('Standard Deviation is ', round(stdev*100, 2), '%')
        st.write('Risk Adjusted Return is ', round(annual_return/(stdev*100), 2))
    except:
        pass

with fundamental_data:
    try:

        # st.write('Fundamental Data')
        key = 'EORDEYI5C4YW0NBE'
        fd = FundamentalData(key, output_format='pandas')
        st.subheader('Balance Sheet')
        balance_sheet = fd.get_balance_sheet_annual(ticker)[0]
        bs = balance_sheet.T[2:]
        bs.columns = list(balance_sheet.T.iloc[0])
        st.write(bs)
        st.subheader('Income Statement')
        income_statement = fd.get_income_statement_annual(ticker)[0]
        is1 = income_statement.T[2:]
        is1.columns = list(income_statement.T.iloc[0])
        st.write(is1)
        st.subheader('Cash Flow Statement')
        cashflow_statement = fd.get_cash_flow_annual(ticker)[0]
        cf = cashflow_statement.T[2:]
        cf.columns = list(cashflow_statement.T.iloc[0])
        st.write(cf)
    except:
        pass

with news:
    try:
        # url = ('https://newsapi.org/v2/everything?q={ticker}&from=2023-04-20&sortBy=popularity&apiKey=420dee08382e423fbc0b67b8148f21a0')
        # r = requests.get(url)
        # df_news = r.json['articles']
        # for i in range(len(df_news)):
        #     st.subheader(f'News {i+1}')
        #     st.write(df_news['title'][i])
        #     st.write(df_news['description'][i])
        # st.write('News')
        # st.header(f'News of {utkarsh_ticker}')
        sn = StockNews(ticker, save_news=True)
        df_news = sn.read_rss()
        for i in range(10):
            st.subheader(f'News {i+1}')
            st.write(df_news['published'][i])
            st.write(df_news['title'][i])
            st.write(df_news['summary'][i])
            title_sentiment = df_news['sentiment_title'][i]
            st.write(f'Title Sentiment {title_sentiment}')
            news_sentiment = df_news['sentiment_summary'][i]
            st.write(f'News Sentiment {news_sentiment}')
    except:
        pass

# session_token = 'eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..L118uUgXhp-arEYE.l8CNWrt1FfI0kxVqk-8-7KoArG8drk9OD9ZIS6WrqK-YHIev5jmmRUBsPesufLk0vJwFxpmSRuNmprI9K1y9ok7MP8P4VLI5buIOqcFgONrKQh2FEndHLCHsT0HY8XWR4pIUaFluZtvDFMZsHg2ksQupP4kWlGdypmzP5-EDnW8BdaZkxiiYuLfQvnF_-VjRyxT8rsU-ucnQAADqkL6xAqn3NZgmB6jbA1AJJDVOd1shd-__21Dl9wP54uE3XJLzqj4TQSIdy8-C4bogtlKnup9D1VE7Wv5ip7ImIKe_-GkyVzhrDeut87S0jldY4O8kNmpXx6f_4OLhxrentIFQ39mpLg_mRwLPsWM29ZXlt9S3vwCFJaqP0tsmEfwEqslTCd_-pvZRIkOFc59z2UC-_4BPCt9cE-so0ZUUz1Hx0bcKGGI79JAjvfN-lxW2MPU03UeBEMi6BopchYlF8qjf1Sco3lC3pPXsrCzbjk_J8XKnTdeyCBRVV7q-lw9z8erDxxZyQRpb5f7SeF7JudMAo19ryMdV6LDFhFU2erKEzhMX9QVuMXm-2XdZk2mdxuJXOaKBPaLz2EyOvuPnrrw2f0TQH6PtwvzKcA1OQGI4QXIVdvIzxoexJPwfUpHUAR8Q00D-q1V0ex0_pBYOlJ7jnPNWu23vg0qqTdW4HAcaXYtJGxPCgrvQ8BTvhPl_Thtj9riKil5DMGH1ETlso0FEI-WFcOWUAm1y64ZapBF-Cpw9jPA6fk8C3_fd5s7JVXlTf1yXyI7BMvkxLhkbSWVoBCP_ps9xqwp3hjLdRrUJMU431_cjK8QgWWfsyGftrATVc16tL6UHnGQb7Umq9BFK6c5q_dmOtYq8R4k0lO8G1YSplfvvWeyBZ_4nYn8r4b6hmt-MZh-R2fTVsiY9e3A4PHD-8SvFF0iammnxHE2VCfX5NSFZntj1A7DZJCFIMMQ8C34YuhZYoFC8_ZqEgFd1YGEZIQal9RMy1NUe5faE4ZSXlB4OitsjoV03hfaZAhww5lXrireORcDML6fVe1o4kgVgdvUMDCYFhCSrjdL9rnxAyIWTdYLCzqTFrfnIU8p7t5a-9ODUCXi_j_xaBUSpLqPVAUikgx4eGk9Vod1L5sunmMPYfdjj4X5aLRjuSv9EUytqm-_CknIO0xQSU8PbX5p-v8z-EsxRVFZdA3vJMH8TeEaRgbqqb0hw4so6lZxCuhaBWyTjYhS2xhgtnYBR-EM-H2Aer0BI1DStNtlzz6vAsHmOwH3uWrsnD_UV0-hq1N77UdNdSjVArps66-eBNQthr68-yco3Ne_4-bZW87JQQNX9tYZZKyIzR8w18KTHTcgw5v15h_O5hrtvY6UbxgapsD8yfjtqSAMbylEZdivNt1chLLRXOVTLzoWDxG3OJYLnhPli7qYkzfoKVQt_UekLWWYzUj7VicZCBslPASYWh1J5XrkLMHt8S0ahKTz8F2W6a0it083ihsbx7b0R9xeqHL2X4f0y7zwEoOPIju1WFkGIGgq6bkinFOe_OuZbeooFLxl3h4asmwxQtuaOCEOhU-lV72YU_yk3QsUPBjrl4PtinKtkfaaEu7_fPnMgOZ77Nd9XvR7VEnZstPthn9Qkl0gksz830o0h5g_v9BWoXQ7TNyUqIySkoehXU9mLvvL3Ma9KTn-2CyONZO051jT46OjlvutKZuTMUGcnjrCd8c_TGU77kqiAHOJvcw6S3qgfTG1ZZylOhsikyktS1Gb-L1HQbYh4DtwqtNuQAekBt7Hor4DbkQzYyyn5RxODuXBjxdZTWNTOcgwOrYl53R56T0iv9c3UwIRuBRlrYmxgExsY7NpjurPNufI3p-HB3FCnWl5_y-WjohbQ7tAuqRvqpFLKD4p19K5LdkYcZSAU4IWQ9Nn6CLvgNAxG44Ho-QSITp7N19u7ov8C-sOj76YVHKQvoSt3YA005salLxYkTQO812-UZAs_JOflG72RpI3i0k4Ff76XyEGW9Rb2dWrCphZvYzw7ZlSm1jD1zsSxbvtvmkncRd5gWsrRdwmBAbtjCNUgO0BuhSyBM6VkIrw_hOQEg8C9C6pCwkRlT7V9wiBRGPnrMZQ-3ktvGP9nYn1RRYZjgm4UHntOiHZgjXHZ8ycfG-CbMyhKBC162eNdWMjwL3nypX3nNZlPXkxY5pDvKLFhSZ3aO1OO3A_KOFsaPZ9iOsqKPOJ3j5DWehx_hG1Gu7FS3anlv2E_dXTs9tekaZVkf2W47_jAQshEJevtsmH9zRHuee5eRGv5ursFRFmHL88tHIjdI7_2GzDQlBvjsETQx0Gj42V6e5NFiIMViZejpOlu7zOcmrwxgUQPd5SWYlFWxq8EM4__QcCz6r6VXeiEBP8hhxlnxefYFSPQ47WSWaPM_nnsThb4W9NnryK2sDfA2hrHe4amXFpM0cp-YfoKM0LF1GEKAETaouM7ZiqxSCuNECPdvke6DjVoCLyjkpz45JXWq02Pw4G8T9QOJgLRCfoII5h7Oy9XNX_n5oNQyXqgOmmSWtc7g7Qo86mcQsJGxPDkwhc2RlIkAXYqG6snRNJQx9tIIIIEjhhLyVCPUlRLisOzY34Z_jXd6ExMMAklZEVLo4CYARIVKa7umgGg7Vtu3dxFp-T7dhNJIam1Pc-97ckbXNA.5D1j_VUJKIOoSLu3lNs4hQ'
# api2 = ChatGPT(session_token)
# buy = api2.send_message(f'3 reasons to buy {ticker} stock')
# sell = api2.send_message(f'3 reasons to sell {ticker} stock')
# swot = api2.send_message(f'SWOT analysis of {ticker} stock')

# with openai1:
#     st.write('Open AI Results')
    # buy_reason, sell_reason, swot_analysis = st.tabs(['3 reasons to buy','3 reasons to sell','SWOT Analysis'])
    # with buy_reason:
    #     st.subheader(f'3 reasons to buy {ticker} stock')
    #     st.write(buy['message'])
    # with sell_reason:
    #     st.subheader(f'3 reasons to sell {ticker} stock')
    #     st.write(sell['message'])
    # with swot_analysis:
    #     st.subheader(f'SWOT Analysis of  {ticker} stock')
    #     st.write(swot['message'])

with tech_indicator:
    st.subheader('Technical Analysis Dashboard')
    df = pd.DataFrame()
    ind_list = df.ta.indicators(as_list=True)
    # st.write(ind_list)
    technical_indicator = st.selectbox('Technical Indicator', options=ind_list)
    method = technical_indicator
    # st.write(getattr(ta,method)(low=data['Low'],close=data['Close'],high=data['High'],open=data['Open'],volume=data['Volume']))
    indicator = pd.DataFrame((getattr(ta, method)(
        low=data['Low'], close=data['Close'], high=data['High'], open=data['Open'], volume=data['Volume'])))
    indicator['Close'] = data['Close']
    figW_ind_new = px.line(indicator)
    st.plotly_chart(figW_ind_new)
    st.write(indicator)


with stocks_comparision:
    st.subheader('Stocks Comparison Dashboard')
    # st.write('Well, Work in Progress. Stay Tuned!')
    # st.write("Pending charts:")
    # st.write("1. Scatter Plot of Stock Returns")
    # st.write("2. Heatmap of Correlations")
    # st.write("3. Stacked Area Chart of Cumulative Returns")

    # select the companies
    col1, col2, = st.columns(2)
    stocks = col1.multiselect('Select the stocks',
                              ["GOOG", "AAPL", "MSFT", "INFY", "TCS.NS"],
                              ['GOOG','AAPL','MSFT'])

    # select the chart
    col2_subcol1, col2_subcol2, col2_subcol3 = col2.columns(3)
    chart_type = ['OHLC', 'Candle Stick Chart', 'Box Plot',
                  'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'All']
    default_ix = chart_type.index('Close')
    sel_chart = col2_subcol1.selectbox(
        'Select the chart', chart_type, index=default_ix)

    # select date range
    today = date.today()
    default_date = today - timedelta(days=2*365)
    start_date_stocks = col2_subcol2.date_input('Start', default_date)
    end_date_stocks = col2_subcol3.date_input('End', today)
    # if end_date not equal to today then add 1 day to end date
    d1 = end_date.strftime('%d-%m-%Y')
    d2 = today.strftime('%d-%m-%Y')
    if d1 != d2:
        end_date_new = end_date_stocks + timedelta(days=1)
    else:
        end_date_new = end_date_stocks

    # 1. Iterate through the stocks and obtain the data and store dataframe in a list
    df_stocks = []
    for stock in stocks:
        df_stocks.append(yf.download(stock, start=start_date_stocks, end=end_date_new))
    # 2. If chart type is not in 'All' or 'OHLC' or 'Candle Stick Chart' or 'Box Plot' then plot the selected chart
    if sel_chart not in ['All', 'OHLC', 'Candle Stick Chart', 'Box Plot']:
        fig = go.Figure()
        for i, df in enumerate(df_stocks):
            fig.add_trace(go.Scatter(
                x=df.index, y=df[sel_chart], name=stocks[i]))
        # Define the slider
        slider = dict(
            active=0,
            steps=[],
            pad=dict(t=50)
        )
        # Define the vertical line shape
        line_shape = dict(type="line", xref="x", yref="paper", x0=start_date_stocks,
                          y0=0, x1=start_date_stocks, y1=1, line=dict(color="red"))

        # Add steps to the slider
        for i in range(len(df.index)):
            slider['steps'].append(dict(
                method="relayout",
                args=[{"shapes[0].x0": df.index[i], "shapes[0].x1": df.index[i]}],
                label=f"{df.index[i].strftime('%Y-%m-%d')}"
            ))

        # Tried to show the intersection value but couldn't figure out how to do it
        # Define the text annotation
        text_annotation = dict(
            x=0.5,
            y=1.05,
            xref='paper',
            yref='paper',
            text='',
            showarrow=False,
            font=dict(size=14),
        )

        fig.layout.update(title_text=f'{sel_chart} Chart', xaxis_rangeslider_visible=False, sliders=[
                          slider], shapes=[line_shape], annotations=[text_annotation])
        # Define the callback function for the slider

        def update_shape(trace, points, state):
            # Get the x-coordinate of the selected point
            selected_date = points.xs[0]
            # Initialize a list to store the intersection values
            intersection_values = []

            # Iterate over all the traces in the figure
            for i in range(len(fig.data)):
                # Get the corresponding y-value for the selected x-coordinate
                y_value = None
                for j in range(len(fig.data[i].x)):
                    if fig.data[i].x[j] == selected_date:
                        y_value = fig.data[i].y[j]
                        break
                # Append the intersection value to the list
                intersection_values.append(y_value)
            # st.write(y_value)
            # Update the x-coordinates of the vertical line shape
            fig.update_layout(shapes=[dict(type="line", xref="x", yref="paper", x0=selected_date, x1=selected_date, y0=0, y1=1, line=dict(color="red"))],
                              annotations=[dict(x=0.5, y=1.05, xref='paper', yref='paper', text=f"Intersection values: {intersection_values}", showarrow=False, font=dict(size=14))])
        # Connect the slider to the callback function
        fig.data[0].on_click(update_shape)
        st.plotly_chart(fig)
    # 3. If chart type is 'OHLC' or 'Candle Stick Chart' or 'Box Plot' then plot the selected chart
    if sel_chart in ['OHLC', 'Candle Stick Chart', 'Box Plot']:
        col1, col2 = st.columns(2)
        fig_new = go.Figure()
        fig_dr = go.Figure()
        for i, df in enumerate(df_stocks):
            if sel_chart == 'OHLC':
                fig_new.add_trace(go.Ohlc(
                    x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=stocks[i]))
            elif sel_chart == 'Candle Stick Chart':
                fig_new.add_trace(go.Candlestick(
                    x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=stocks[i]))
            elif sel_chart == 'Box Plot':
                df["Daily_Return"] = (df["Close"] - df["Close"].shift(1))
                fig_dr.add_trace(go.Box(y=df["Daily_Return"].dropna(),
                                        name=stocks[i]))

                fig_new.add_trace(
                    go.Box(y=df['Close'].dropna(), name=stocks[i]))
                # break

        if sel_chart == 'Box Plot':
            fig_dr.update_layout(title="Daily Returns",
                                 yaxis_title="Return")
            fig_new.update_layout(title_text=f'{sel_chart} For Close Value',
                                  xaxis_rangeslider_visible=False, xaxis_title="Date", yaxis_title="Price")
            col1.plotly_chart(fig_dr, use_container_width=True)
            col2.plotly_chart(fig_new, use_container_width=True)
        else:
            fig_new.update_layout(
                title_text=f'{sel_chart} Chart', xaxis_rangeslider_visible=False, xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig_new)
    # 4. If chart type is 'All' then plot all the charts for ['Candle Stick Chart','Box Plot','Open', 'High', 'Low', 'Close','Adj Close', 'Volume'] separately with two charts eaach in a row
    if sel_chart == 'All':
        col1, col2 = st.columns(2)
        fig_ohlc = go.Figure()
        fig_cs = go.Figure()
        fig_bp = go.Figure()
        fig_dr = go.Figure()
        fig_open = go.Figure()
        fig_close = go.Figure()
        fig_high = go.Figure()
        fig_low = go.Figure()
        fig_vol = go.Figure()
        for i, df in enumerate(df_stocks):
            fig_ohlc.add_trace(go.Ohlc(
                x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=stocks[i]))
            fig_cs.add_trace(go.Candlestick(
                x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=stocks[i]))
            fig_bp.add_trace(go.Box(y=df['Close'].dropna(), name=stocks[i]))
            df["Daily_Return"] = (df["Close"] - df["Close"].shift(1))
            fig_dr.add_trace(go.Box(y=df["Daily_Return"].dropna(),
                                    name=stocks[i]))
            fig_open.add_trace(go.Scatter(
                x=df.index, y=df['Open'], name=stocks[i]))
            fig_close.add_trace(go.Scatter(
                x=df.index, y=df['Close'], name=stocks[i]))
            fig_high.add_trace(go.Scatter(
                x=df.index, y=df['High'], name=stocks[i]))
            fig_low.add_trace(go.Scatter(
                x=df.index, y=df['Low'], name=stocks[i]))
            fig_vol.add_trace(go.Scatter(
                x=df.index, y=df['Volume'], name=stocks[i]))
        fig_ohlc.update_layout(
            title_text='OHLC Chart', xaxis_rangeslider_visible=False, xaxis_title="Date", yaxis_title="Price")
        fig_cs.update_layout(title_text='Candle Stick Chart',
                             xaxis_rangeslider_visible=False, xaxis_title="Date", yaxis_title="Price")
        fig_bp.update_layout(title_text='Box Plot For Close Value',
                             xaxis_rangeslider_visible=False, xaxis_title="Company", yaxis_title="Price")
        fig_dr.update_layout(title="Daily Returns",
                             xaxis_title="Company", yaxis_title="Return")
        fig_open.update_layout(
            title="Open Chart", xaxis_title="Date", yaxis_title="Price")
        fig_close.update_layout(title="Close Chart",
                                xaxis_title="Date", yaxis_title="Price")
        fig_high.update_layout(
            title="High Chart", xaxis_title="Date", yaxis_title="Price")
        fig_low.update_layout(
            title="Low Chart", xaxis_title="Date", yaxis_title="Price")
        fig_vol.update_layout(title="Volume Chart",
                              xaxis_title="Volume", yaxis_title="Volume")
        col1.plotly_chart(fig_ohlc, use_container_width=True)
        col2.plotly_chart(fig_cs, use_container_width=True)
        col1.plotly_chart(fig_bp, use_container_width=True)
        col2.plotly_chart(fig_dr, use_container_width=True)
        col1.plotly_chart(fig_open, use_container_width=True)
        col2.plotly_chart(fig_close, use_container_width=True)
        col1.plotly_chart(fig_high, use_container_width=True)
        col2.plotly_chart(fig_low, use_container_width=True)
        st.plotly_chart(fig_vol, use_container_width=True)

    # Scatter Plot of Stock Returns
    st.subheader("Scatter Plot of Stock Returns")
    for df_stock in df_stocks:
        # df_stock["Daily_Return"] = (df_stock["Close"] - df_stock["Close"].shift(1))
        df_stock["Return"] = df_stock["Close"].pct_change()
        df_stock["Cumulative_Return"] = (1 + df_stock["Return"]).cumprod()
    # Combine the returns for all stocks into one dataframe
    df_returns = pd.concat([df_stock["Return"] for df_stock in df_stocks], axis=1)
    df_returns.columns = stocks
    fig = px.scatter_matrix(df_returns, width=500, height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    # Heatmap of Correlations
    col1.subheader("Heatmap of Correlations")
    # Calculate the correlation matrix
    corr = df_returns.corr()

    # Create a heatmap of the correlations
    fig = px.imshow(corr,
                    labels=dict(x="Stocks", y="Stocks", color="Correlation"),
                    x=df_returns.columns,
                    y=df_returns.columns,
                    color_continuous_scale="RdBu")

    # Update the figure layout
    fig.update_layout(title="Correlation Matrix of Daily Returns",
                    width=500,
                    height=500)
    col1.plotly_chart(fig, use_container_width=True)
    
    
    # Stacked Area Chart of Cumulative Returns
    col2.subheader("Stacked Area Chart of Cumulative Returns")
    # Combine the cumulative returns for all stocks into one dataframe
    df_cumulative_returns = pd.concat([df_stock["Cumulative_Return"] for df_stock in df_stocks], axis=1)
    df_cumulative_returns.columns = stocks
    # Create a stacked area chart of the cumulative returns
    fig = go.Figure()
    for col in df_cumulative_returns.columns:
        fig.add_trace(go.Scatter(x=df_cumulative_returns.index, y=df_cumulative_returns[col],
                                mode="lines", stackgroup="one", name=col))

    # Set the chart title and axes labels
    fig.update_layout(title="Cumulative Returns of Multiple Stocks",
                    xaxis_title="Date",
                    yaxis_title="Cumulative Return")
    col2.plotly_chart(fig, use_container_width=True)
    # df = yf.download(stocks, start=start_date_stocks, end=end_date_new)
    # st.write(df)


with stock_prediction:
    st.header('Stock Prediction Dashboard')
    
    #Load Model
    @st.cache_resource
    def load_lstm_model(model_name):
        model = load_model(model_name)
        st.success("Loaded LSTM model")  # 👈 Show a success message
        return model
    
    # yf.pdr_override()
    
    col1, col2, col3,col4 = st.columns(4)
    # select the companies
    stock_type = ["GOOG", "AAPL", "MSFT", "INFY", "TCS.NS", "GME"]
    default_ix = stock_type.index('GOOG')
    sel_chart = col1.selectbox(
        'Select the stock', stock_type, index=default_ix)
    
    # select date range
    today = date.today()
    default_date = today - timedelta(days=1*365)
    start_date_stocks = '2010-01-01'
    investment_date = col2.date_input('Investment Date', default_date)
    # start_date_stocks = col2.date_input(
    #     'Past Date for Training ', default_date)
    # Give the investment amount
    investment = col3.number_input('Inv. Amount', min_value=100, step=1)
    n_years = col1.slider('Years of prediction:', 1, 5)
    # Give a future date to predict the value
    future_date = col4.date_input('Prediction Date', value=today+timedelta(
        days=1), min_value=investment_date, max_value=today+timedelta(days=n_years*365-1))
    col4.info('Valid Future Date for LSTM is till '+str((today+timedelta(days=100)).strftime('%d-%m-%Y')), icon="ℹ️")
    data = yf.download(sel_chart, start_date_stocks, today+timedelta(days=1))
    # data.reset_index(inplace=True)
    #Code for LSTM Prediction
    st.subheader("Prediction using LSTM")
    model=load_lstm_model('lstm_prediction_stock.h5')
    scale_value=max(data['Close'])
    scaling_value=1/scale_value
    #Reset Index
    dates=data.index
    input_data=data[['Close']]*scaling_value
    input_data=input_data.values
    x_test,y_test=[],[]
    days=100
    for i in range(days,input_data.shape[0]):
        x_test.append(input_data[i-days:i])
        y_test.append(input_data[i,0])
    
    x_test,y_test=np.array(x_test),np.array(y_test)
    #Prediction
    y_pred=model.predict(x_test)
    #Re-Scaling Y-pred
    y_pred=y_pred*scale_value
    y_pred=pd.DataFrame(y_pred,columns=['Close'])
    y_pred=y_pred.set_index(dates[100:])
    final_df=data
    final_df['Prediction']=y_pred['Close']
    final_df['Date']=final_df.index
    final_df['Original']=final_df['Close']
    #Future Prediction using LSTM Model
    initial_data=data.tail(100)[['Close']]
    input_data=initial_data['Close']*scaling_value
    input_data=input_data.values
    sliding_window=list(input_data)
    prediction_list=[]
    # date_object = datetime.strptime(today, '%Y-%m-%d').date()
    date_object = today
    # days=(date_object-data.index[-1].to_pydatetime().date()).days
    # days=max(days,0)
    # days=min(days,150)
    days = 100
    for i in range(days):
        input_data=np.array([np.array(sliding_window).reshape(-1,1)])
        y_pred=model.predict(input_data)
        prediction_list.append(y_pred[0][0])
        del sliding_window[0]
        sliding_window.append(prediction_list[-1])
    predictions=np.array(prediction_list)*scale_value
    new_dates=np.array([df.index[-1].to_pydatetime().date()+timedelta(days=x) for x in range(1,days+1)])
    new_df=pd.DataFrame(predictions,columns=['Prediction'])
    new_df=new_df.set_index(new_dates)
    final_df=final_df.append(new_df)
    final_df['Date']=final_df.index
    fig = px.line(final_df, x='Date', y=["Original","Prediction"])
    fig.update_layout(
        title="Stock Prediction",
        xaxis_title="Date",
        yaxis_title="Stock Price",
        legend_title="Type",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    
    if investment_date.strftime("%Y-%m-%d") not in final_df['Date'].dt.strftime("%Y-%m-%d").values:
            nearest_date = final_df[final_df['Date'] < investment_date.strftime("%Y-%m-%d")]['Date'].dt.strftime("%Y-%m-%d") .values[-1]
    else:
        nearest_date = investment_date.strftime("%Y-%m-%d")

    if future_date.strftime("%Y-%m-%d") not in final_df['Date'].dt.strftime("%Y-%m-%d").values:
            nearest_future_date = final_df[final_df['Date'] < future_date.strftime("%Y-%m-%d")]['Date'].dt.strftime("%Y-%m-%d").values[-1]
    else:
        nearest_future_date = future_date.strftime("%Y-%m-%d")
    
    # nearest_date = final_df['Date'][final_df['Date'].get_loc(investment_date, method='nearest')]
    # nearest_future_date = final_df['Date'][final_df['Date'].get_loc(future_date, method='nearest')]
    today_value = final_df[final_df['Date'] == nearest_date]['Original'].values[0]
    future_value = final_df[final_df['Date'] ==
                            nearest_future_date]['Prediction'].values[0]
    # calculate percentage change
    percentage_change = (future_value - today_value)/today_value*100
    # calculate investment value on future date using percentage change
    future_investment_value = round(investment*(1+percentage_change/100), 2)
    # display the percentage change and future investment value
    with col2:
        st.markdown("<span style='color:blue;text-decoration:underline;'>For LSTM model</span>", unsafe_allow_html=True)
        st.write('Nearest Date: ',nearest_date)
        st.write("Nearest F-Date: ",nearest_future_date)
        if future_date.strftime('%d-%m-%Y') > (today+timedelta(days=100)).strftime('%d-%m-%Y'):
            st.markdown(f'<span style="color:red"> Future Date '+future_date.strftime('%d-%m-%Y')+' is outside the valid date for LSTM </span>', unsafe_allow_html=True)
        else:
            st.markdown(f'Investment Stock value: <span style="color:green">' +
                        str(round(today_value, 2))+'</span>', unsafe_allow_html=True)
            original_value = final_df[final_df['Date'] == nearest_future_date]['Original'].values[0]
            # st.write(original_value)
            # check if original value is nan
            if pd.isnull(original_value):
                original_value = 'color:red">NA'
            else:
                original_value = 'color:green">'+str(round(original_value, 2))
            st.markdown(f'Actual Stock value: <span style="' +
                        original_value+'</span>', unsafe_allow_html=True)
                
            if future_value < today_value:
                color = "red"
            else:
                color = "green"

            st.markdown(f'Predicted Stock value: <span style="color:{color}">'+str(
                round(future_value, 2))+'</span>', unsafe_allow_html=True)
            st.markdown(f'% change in value: <span style="color:{color}">'+str(
                round(percentage_change, 2))+'%</span>', unsafe_allow_html=True)
            st.markdown(f'Investment value: <span style="color:green">' +
                        str(investment)+'</span>', unsafe_allow_html=True)
            st.markdown(f'Pred. Investment Val: <span style="color:{color}">' + str(
                future_investment_value)+'</span>', unsafe_allow_html=True)
    
    #Code for Prophet prediction
    st.subheader("Prediction Using Prophet")
    period = n_years * 365
    
   
    # Plot raw data
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['Date'], y=data['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(
            x=data['Date'], y=data['Close'], name="stock_close"))
        fig.layout.update(
            title_text='Time Series data for '+sel_chart, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    # Predict forecast with Prophet.
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)
    # check condition if future value is within the prediction range.
    # get Close value on today's date(if not available get nearest) and calculate the percentage change, Use it to predict the future value.
    # Check if today's date is present in the "Date" column of data. If not get the nearest date to todays's date.
    if investment_date.strftime("%Y-%m-%d") not in data['Date'].dt.strftime("%Y-%m-%d").values:
        nearest_date = data[data['Date'] < investment_date.strftime(
            "%Y-%m-%d")]['Date'].dt.strftime("%Y-%m-%d").values[-1]
    else:
        nearest_date = investment_date.strftime("%Y-%m-%d")
    if future_date.strftime("%Y-%m-%d") not in forecast['ds'].dt.strftime("%Y-%m-%d").values:
        nearest_future_date = forecast[forecast['ds'] < future_date.strftime("%Y-%m-%d")]['ds'].dt.strftime("%Y-%m-%d").values[-1]
    else:
        nearest_future_date = future_date.strftime("%Y-%m-%d")
    
    today_value = data[data['Date'] == nearest_date]['Close'].values[0]
    future_value = forecast[forecast['ds'] ==
                            nearest_future_date]['yhat'].values[0]
    # calculate percentage change
    percentage_change = (future_value - today_value)/today_value*100
    # calculate investment value on future date using percentage change
    future_investment_value = round(investment*(1+percentage_change/100), 2)
    # display the percentage change and future investment value
    with col3:
        st.markdown("<span style='color:blue;text-decoration:underline;'>For Prophet model</span>", unsafe_allow_html=True)
        st.write('Nearest Date: ',nearest_date)
        st.write("Nearest F-Date: ",nearest_future_date)
        # st.write("Current Stock value: ",round(today_value,2))
        st.markdown(f'Investment Stock value: <span style="color:green">' +
                    str(round(today_value, 2))+'</span>', unsafe_allow_html=True)
        try:
            original_value = data[data['Date'] == nearest_future_date]['Close'].values[0]
            original_value = 'color:green">'+str(round(original_value, 2))
        except:
            original_value = 'color:red">NA'
            
        # if original_value is pd.NA:
        #     original_value = 'color:red">Not Known'
        # else:
        #     original_value = 'color:green">'+str(round(original_value, 2))
        st.markdown(f'Actual Stock value: <span style="' +
                    original_value+'</span>', unsafe_allow_html=True)
        if future_value < today_value:
            color = "red"
        else:
            color = "green"

        st.markdown(f'Future Stock value: <span style="color:{color}">'+str(
            round(future_value, 2))+'</span>', unsafe_allow_html=True)
        st.markdown(f'% change in value: <span style="color:{color}">'+str(
            round(percentage_change, 2))+'%</span>', unsafe_allow_html=True)
        st.markdown(f'Investment value: <span style="color:green">' +
                    str(investment)+'</span>', unsafe_allow_html=True)
        st.markdown(f'Pred. Investment Val: <span style="color:{color}">' + str(
            future_investment_value)+'</span>', unsafe_allow_html=True)
    # st.write(forecast)
    # future_dates = []
    # future_predicted_values = []
    # # iterate from today's date to the end of forecast dataframe dates and calculate the future investment value on every date.Add the date and future investment value to the corresponding lists.
    # for i in range(len(forecast['ds'])):
    #     if forecast['ds'][i] > investment_date:
    #         # convert the dates in forecast to string and append to the list
    #         future_dates.append((forecast['ds'][i]).strftime("%Y-%m-%d"))
    #         # future_dates.append((forecast['ds'][i]))
    #         future_predicted_values.append(
    #             round(investment*(1+((forecast['yhat'][i]-today_value)/today_value*100)/100), 2))

    # # Determine the index of the date you want to highlight
    # highlight_index = future_dates.index(nearest_future_date)
   
    # # Create a new list of values with the highlighted value
    # highlighted_values = [None] * len(future_predicted_values)
    # highlighted_values[highlight_index] = future_predicted_values[highlight_index]

    # fig = go.Figure()
    # fig.add_trace(go.Scatter(
    #     x=future_dates, y=future_predicted_values, mode='lines', name='Pred. Values'))

    # # Add the highlighted value trace
    # fig.add_trace(go.Scatter(x=future_dates, y=highlighted_values, mode='markers',
    #               marker=dict(color='red', size=10), name='Highlighted Value'))

    # # Set the chart title and axis labels

    # fig.update_layout(title='Investment Predictions over Time',
    #                   xaxis_title='Future Date', yaxis_title='Predicted Investment Value')
    # st.plotly_chart(fig, use_container_width=True)
    
    # Show and plot forecast
    agree = st.checkbox('Show Forecast Data')
    if agree:
        st.subheader('Forecast data')
        st.write(forecast.head())
        st.write(forecast.tail())

    fig1 = plot_plotly(m, forecast)
    fig1.layout.update(
        title_text=f'Forecast plot for {n_years} years', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig1, user_container_width=True)

    #LSTM Prediction
    

with portfolio_optimization:
    st.subheader('Portfolio Optimization Dashboard')
    # st.write('Hold On, we are still working on this section.')
    col1, col2, = st.columns(2)
    stocks = col1.multiselect('Select the stocks',
                              ["GOOG", "AAPL", "MSFT", "INFY", "TCS.NS","AMZN","TSLA","META","INFY"],
                              ["GOOG","AAPL","MSFT","TSLA","META"])
    investment_amount = col2.number_input('Investment Amount', min_value=1000, step=100)
    end = datetime.today()
    start = end - timedelta(days=2000)
    # stocks = ['AMZN','TSLA','META','INFY']
    no_companies = len(stocks)
    #Combine all the data into a single dataframe
    df = yf.download(stocks[0],start,end)[['Close']]
    df.columns = [stocks[0]]
    for stock in stocks[1:]:
        df[stock] = yf.download(stock,start,end)['Close']
    df.dropna(inplace=True)
    fig = px.line(df * 100 / df.iloc[0])
    col1.plotly_chart(fig, use_container_width=True)
    np.random.seed(1)
    # Weight each security
    weights = np.random.random((no_companies,1))
    # normalize it, so that some is one
    weights /= np.sum(weights)
    # st.write(f'Normalized Weights : {weights.flatten()}')

    # We generally do log return instead of return
    log_ret = np.log(df / df.shift(1))

    # Expected return (weighted sum of mean returns). Mult by 252 as we always do annual calculation and year has 252 business days
    exp_ret = log_ret.mean().dot(weights)*252 
    # st.write(f'\nExpected return of the portfolio is : {exp_ret[0]}')

    # Exp Volatility (Risk)
    exp_vol = np.sqrt(weights.T.dot(252*log_ret.cov().dot(weights)))
    # st.write(f'\nVolatility of the portfolio: {exp_vol[0][0]}')

    # Sharpe ratio
    sr = exp_ret / exp_vol
    # st.write(f'\nSharpe ratio of the portfolio: {sr[0][0]}')
    
    #Monte Carlo Simulation
    # number of simulation
    n = 50_000
    # n = 10

    port_weights = np.zeros(shape=(n,len(df.columns)))
    port_volatility = np.zeros(n)
    port_sr = np.zeros(n)
    port_return = np.zeros(n)

    num_securities = len(df.columns)
    # num_securities
    for i in range(n):
        # Weight each security
        weights = np.random.random(no_companies)
        # normalize it, so that some is one
        weights /= np.sum(weights)
        port_weights[i,:] = weights 
        #     print(f'Normalized Weights : {weights.flatten()}')

        # Expected return (weighted sum of mean returns). Mult by 252 as we always do annual calculation and year has 252 business days
        exp_ret = log_ret.mean().dot(weights)*252 
        port_return[i] = exp_ret
    #     print(f'\nExpected return is : {exp_ret[0]}')

        # Exp Volatility (Risk)
        exp_vol = np.sqrt(weights.T.dot(252*log_ret.cov().dot(weights)))
        port_volatility[i] = exp_vol
    #     print(f'\nVolatility : {exp_vol[0][0]}')

        # Sharpe ratio
        sr = exp_ret / exp_vol
        port_sr[i] = sr
    #     print(f'\nSharpe ratio : {sr[0][0]}')
    # Index of max Sharpe Ratio
    max_sr = port_sr.max()
    ind = port_sr.argmax()
    # Return and Volatility at Max SR
    max_sr_ret = port_return[ind]
    max_sr_vol = port_volatility[ind]
    
    fig1 = plt.figure(figsize=(20,15))
    plt.scatter(port_volatility,port_return,c=port_sr, cmap='plasma')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility', fontsize=15)
    plt.ylabel('Return', fontsize=15)
    plt.title('Efficient Frontier (Bullet Plot)', fontsize=15)
    plt.scatter(max_sr_vol, max_sr_ret, c='blue', s=150, edgecolors='red', marker='o', label='Max \
    Sharpe ratio Portfolio')
    plt.legend()
    col2.pyplot(fig1, use_container_width=True)
    col1,col2 = st.columns(2)
    investment_amount_distribution = [investment_amount * weight for weight in port_weights[ind]]
    for weight, stock in zip(investment_amount_distribution,stocks):
        col1.write(f'{round(weight, 2)} of {stock} should be bought.')
    # for weight, stock in zip(port_weights[ind],stocks):
    #     st.write(f'{round(weight * 100, 2)} % of {stock} should be bought.')
        
    # best portfolio return
    col1.write(f'\nMarkowitz optimal portfolio return is : {round(max_sr_ret * 100, 2)}% with volatility \
    {max_sr_vol}')
    pie_chart=pd.DataFrame(port_weights[ind]*100,index=stocks,columns=['weight'])
    pie_chart['company']=stocks
    fig2 = px.pie(pie_chart, values='weight', names='company', title='Portfolio Weights')
    col2.plotly_chart(fig2, use_container_width=True)