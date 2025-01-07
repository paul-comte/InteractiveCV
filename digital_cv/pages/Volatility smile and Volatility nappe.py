import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
import matplotlib.pyplot as plt

def download_all_options(ticker):
    stock = yf.Ticker(ticker)
    options_dates = stock.options
    all_options = []
    S0 = yf.download(tickers=ticker, start='2025-01-01', end='2025-01-07')['Close'].iloc[-1]
    S0 = S0.values[0]
    for date in options_dates:
        opt = stock.option_chain(date)
        calls = opt.calls
        puts = opt.puts
        expiration_date = datetime.strptime(date, '%Y-%m-%d')
        timedelta = (expiration_date - datetime.today()).days
        calls['expirationDate'] = timedelta
        puts['expirationDate'] = timedelta
        calls['optionType'] = 'call'
        puts['optionType'] = 'put'
        all_options.append(calls)
        all_options.append(puts)

    all_options_df = pd.concat(all_options, ignore_index=True)
    all_options_df["midPrice"] = (all_options_df["ask"] + all_options_df["bid"]) / 2
    all_options_df["strike"] = all_options_df["strike"].astype(float)
    all_options_df = all_options_df[(S0 * 0.8 < all_options_df["strike"]) & (all_options_df["strike"] < S0 * 1.2)]
    all_options_df = all_options_df[["expirationDate", "strike", "midPrice", "optionType"]]

    return all_options_df.pivot_table(index=['expirationDate','strike'],
                                          columns='optionType',
                                          values='midPrice')

def BSM(sigma, S0, K, P, T, r, option):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option == 'call':
        result = (S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)) - P
    if option == 'put':
        result = (K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)) - P
    return result

def plot_volatility_smile(aapl_options, S0, r):
    imp_vols = []
    option_type = "call"
    for index, value in aapl_options["call"].items():
        T = index[0] / 365
        K = index[1]
        sigma0 = 0.5
        imp_vol = fsolve(BSM, sigma0, args=(S0, K, value, T, r, option_type), xtol=1e-6, maxfev=500)
        imp_vols.append(float(imp_vol))

    maturities = aapl_options[option_type].index.levels[0]
    day1 = aapl_options[option_type].index.levels[0][0]
    day2 = aapl_options[option_type].index.levels[0][2]
    day3 = aapl_options[option_type].index.levels[0][4]
    day4 = aapl_options[option_type].index.levels[0][6]
    day5 = aapl_options[option_type].index.levels[0][7]

    imp_vols_indexed = pd.Series(imp_vols, index=aapl_options[option_type].index)

    plt.figure(figsize=(12, 8))
    plt.title('Volatility Smile')

    plt.plot(pd.Series(imp_vols, index=aapl_options[option_type].index).loc[day1], label=f'Maturity in {day1} days')
    plt.plot(pd.Series(imp_vols, index=aapl_options[option_type].index).loc[day2], label=f'Maturity in {day2} days')
    plt.plot(pd.Series(imp_vols, index=aapl_options[option_type].index).loc[day3], label=f'Maturity in {day3} days')
    plt.plot(pd.Series(imp_vols, index=aapl_options[option_type].index).loc[day4], label=f'Maturity in {day4} days')
    plt.plot(pd.Series(imp_vols, index=aapl_options[option_type].index).loc[day5], label=f'Maturity in {day5} days')

    plt.legend()
    st.pyplot(plt)

def plot_volatility_surface(aapl_options, S0, r):
    imp_vols = []
    option_type = "call"
    for index, value in aapl_options["call"].items():
        T = index[0] / 365
        K = index[1]
        sigma0 = 0.5
        imp_vol = fsolve(BSM, sigma0, args=(S0, K, value, T, r, option_type), xtol=1e-6, maxfev=500)
        imp_vols.append(float(imp_vol))

    imp_vols_indexed = pd.Series(imp_vols, index=aapl_options[option_type].index)
    df_interpolated = imp_vols_indexed.unstack(0).interpolate(method='linear')

    x = np.array(aapl_options[option_type].index.levels[0])
    y = np.array(aapl_options[option_type].index.get_level_values(1).unique())
    z = np.array(df_interpolated)

    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
    fig.update_layout(title='Volatility Surface', autosize=False,
                      width=700, height=700,
                      margin=dict(l=65, r=50, b=65, t=90))
    fig.update_scenes(xaxis_title='Maturity (days until strike date)', yaxis_title='Strike', zaxis_title='Implied Volatility')

    st.plotly_chart(fig)

def main():
    st.title("Volatility Smile and Surface Visualization")

    st.markdown(
        """
        <div style="background-color:#ff7f00;padding:10px;border-radius:5px;">
            <strong style="color:#721c24;">Disclaimer:</strong>
            <p style="color:#721c24;">
                I am actually using yfinance to download the options data. It may lead to some discrepancies in the results.
                As it is a free API, it may not be as accurate as a paid one.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    ticker = st.text_input("Enter the stock ticker (e.g., AAPL):")
    r = st.number_input("Enter the risk-free rate (e.g., 0.01 for 1%):", value=0.01)

    if st.button("Generate Plots"):
        if ticker:
            aapl_options = download_all_options(ticker)
            S0 = yf.download(tickers=ticker, start='2025-01-01', end='2025-01-07')['Close'].iloc[-1]
            S0 = S0.values[0]

            st.write(f"Current stock price (S0): {S0}")

            st.write("### Volatility Smile")
            plot_volatility_smile(aapl_options, S0, r)

            st.write("### Volatility Surface")
            plot_volatility_surface(aapl_options, S0, r)
        else:
            st.error("Please enter a valid stock ticker.")

if __name__ == "__main__":
    main()

st.subheader("Source code")
code = """
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
import matplotlib.pyplot as plt

def download_all_options(ticker):
    stock = yf.Ticker(ticker)
    options_dates = stock.options
    all_options = []
    S0 = yf.download(tickers=ticker, start='2025-01-01', end='2025-01-07')['Close'].iloc[-1]
    S0 = S0.values[0]
    for date in options_dates:
        opt = stock.option_chain(date)
        calls = opt.calls
        puts = opt.puts
        expiration_date = datetime.strptime(date, '%Y-%m-%d')
        timedelta = (expiration_date - datetime.today()).days
        calls['expirationDate'] = timedelta
        puts['expirationDate'] = timedelta
        calls['optionType'] = 'call'
        puts['optionType'] = 'put'
        all_options.append(calls)
        all_options.append(puts)

    all_options_df = pd.concat(all_options, ignore_index=True)
    all_options_df["midPrice"] = (all_options_df["ask"] + all_options_df["bid"]) / 2
    all_options_df["strike"] = all_options_df["strike"].astype(float)
    all_options_df = all_options_df[(S0 * 0.8 < all_options_df["strike"]) & (all_options_df["strike"] < S0 * 1.2)]
    all_options_df = all_options_df[["expirationDate", "strike", "midPrice", "optionType"]]

    return all_options_df.pivot_table(index=['expirationDate','strike'],
                                          columns='optionType',
                                          values='midPrice')

def BSM(sigma, S0, K, P, T, r, option):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option == 'call':
        result = (S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)) - P
    if option == 'put':
        result = (K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)) - P
    return result

def plot_volatility_smile(aapl_options, S0, r):
    imp_vols = []
    option_type = "call"
    for index, value in aapl_options["call"].items():
        T = index[0] / 365
        K = index[1]
        sigma0 = 0.5
        imp_vol = fsolve(BSM, sigma0, args=(S0, K, value, T, r, option_type), xtol=1e-6, maxfev=500)
        imp_vols.append(float(imp_vol))

    maturities = aapl_options[option_type].index.levels[0]
    day1 = aapl_options[option_type].index.levels[0][0]
    day2 = aapl_options[option_type].index.levels[0][2]
    day3 = aapl_options[option_type].index.levels[0][4]
    day4 = aapl_options[option_type].index.levels[0][6]
    day5 = aapl_options[option_type].index.levels[0][7]

    imp_vols_indexed = pd.Series(imp_vols, index=aapl_options[option_type].index)

    plt.figure(figsize=(12, 8))
    plt.title('Volatility Smile')

    plt.plot(pd.Series(imp_vols, index=aapl_options[option_type].index).loc[day1], label=f'Maturity in {day1} days')
    plt.plot(pd.Series(imp_vols, index=aapl_options[option_type].index).loc[day2], label=f'Maturity in {day2} days')
    plt.plot(pd.Series(imp_vols, index=aapl_options[option_type].index).loc[day3], label=f'Maturity in {day3} days')
    plt.plot(pd.Series(imp_vols, index=aapl_options[option_type].index).loc[day4], label=f'Maturity in {day4} days')
    plt.plot(pd.Series(imp_vols, index=aapl_options[option_type].index).loc[day5], label=f'Maturity in {day5} days')

    plt.legend()
    st.pyplot(plt)

def plot_volatility_surface(aapl_options, S0, r):
    imp_vols = []
    option_type = "call"
    for index, value in aapl_options["call"].items():
        T = index[0] / 365
        K = index[1]
        sigma0 = 0.5
        imp_vol = fsolve(BSM, sigma0, args=(S0, K, value, T, r, option_type), xtol=1e-6, maxfev=500)
        imp_vols.append(float(imp_vol))

    imp_vols_indexed = pd.Series(imp_vols, index=aapl_options[option_type].index)
    df_interpolated = imp_vols_indexed.unstack(0).interpolate(method='linear')
"""
st.code(code, language="python")