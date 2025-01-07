import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import altair as alt
from datetime import datetime

# --------------------------
#        STOCK CLASS
# --------------------------
class Stock:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = None   # Will store the raw historical DataFrame
        self.returns = None  # Prepared DataFrame with Date, Price, log_ret
    
    def calculate_stock_prices(self, start):
        """
        Downloads data from 'start' until the most recent date available.
        Renames 'Close' to 'Price'.
        No more tail(200) restriction, so we keep the entire dataset.
        """
        data = yf.download(self.ticker, start=start)
        if data.empty:
            return None
        
        # Keep only 'Close', rename to 'Price'
        data = data[['Close']].reset_index()
        data.rename(columns={'Close': 'Price'}, inplace=True)
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        
        # Must have at least 2 points to plot a line
        if len(data) < 2:
            return None
        
        # Store the full dataset in self.data and self.returns
        self.data = data.copy()
        self.returns = data.copy()
        
        return self.returns
    
    def calculate_log_returns(self):
        """
        Computes log returns (log_ret) on the 'Price' column
        and stores it in self.returns['log_ret'].
        """
        if self.returns is None or self.returns.empty:
            return None
        
        self.returns['log_ret'] = (
            np.log(self.returns['Price']) 
            - np.log(self.returns['Price'].shift(1))
        )
        return self.returns['log_ret']
    
    def create_simulation_data(self, number_sims, duration):
        """
        Returns a DataFrame containing only Monte Carlo simulations.
        Each simulation has columns: 'Date', 'Price', and 'Simulation'
        labeled as 'Simulation 1', 'Simulation 2', etc.
        """
        if (self.returns is None) or (self.returns.empty):
            return None
        
        if 'log_ret' not in self.returns.columns or self.returns['log_ret'].dropna().empty:
            return None
        
        # Statistics on log returns
        mean_ret = self.returns['log_ret'].mean()
        std_ret = self.returns['log_ret'].std()
        
        # Last historical price to start simulations
        last_price = float(self.returns['Price'].iloc[-1])
        # Last historical date
        last_date = pd.to_datetime(self.returns['Date'].iloc[-1])
        
        sim_list = []
        for i in range(number_sims):
            sim_rets = np.random.normal(mean_ret, std_ret, duration)
            sim_prices = last_price * (sim_rets + 1).cumprod()
            
            sim_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=duration, 
                freq='D'
            )
            
            df_sim = pd.DataFrame({
                'Date': sim_dates,
                'Price': sim_prices,
                'Simulation': f"Simulation {i+1}"
            })
            sim_list.append(df_sim)
        
        sims_df = pd.concat(sim_list, ignore_index=True)
        return sims_df

# --------------------------
#    STREAMLIT APPLICATION
# --------------------------
def main():
    st.title("Historical Data & Monte Carlo Simulation")

    # 1. Ticker input
    ticker_input = st.text_input("Enter the ticker (e.g. AAPL, TSLA...)", value="AAPL")
    
    # 2. Predefined start dates (no end date => yfinance retrieves until the most recent)
    #    Extended options back to the year 2000:
    period_options = {
        "Since 2000-01-01": "2000-01-01",
        "Since 2005-01-01": "2005-01-01",
        "Since 2010-01-01": "2010-01-01",
        "Since 2015-01-01": "2015-01-01",
        "Since 2019-01-01": "2019-01-01",
        "Since 2022-01-01": "2022-01-01"
    }
    chosen_period = st.selectbox("Choose the start date:", list(period_options.keys()))
    start_date = period_options[chosen_period]
    
    # 3. Monte Carlo parameters
    number_sims = st.number_input(
        "Number of simulations",
        min_value=1, max_value=1000, value=30, step=1
    )
    duration = st.number_input(
        "Simulation duration (days)",
        min_value=1, max_value=5000, value=252, step=1
    )
    
    # 4. Run simulation
    if st.button("Run Simulation"):
        if not ticker_input.strip():
            st.error("Please enter a valid ticker.")
            return
        
        stock = Stock(ticker_input.strip().upper())
        
        # a) Download historical data
        df_prices = stock.calculate_stock_prices(start_date)
        if df_prices is None:
            st.error("No data or not enough data for this ticker/period.")
            return
        
        # b) Compute log returns
        log_rets = stock.calculate_log_returns()
        if log_rets is None or log_rets.empty:
            st.error("Failed to compute log returns. Check the data.")
            return
        
        # c) Create simulation DataFrame
        sims_df = stock.create_simulation_data(number_sims, duration)
        if sims_df is None or sims_df.empty:
            st.error("An error occurred while generating simulations.")
            return
        
        # --------------------------
        #  1) HISTORICAL CHART
        # --------------------------
        hist_df = stock.data.copy()  # Contains Date, Price
        hist_chart = alt.Chart(hist_df).mark_line(color='blue').encode(
            x=alt.X('Date:T', title='Date'),
            y=alt.Y('Price:Q', title='Price'),
            tooltip=['Date:T', 'Price:Q']
        ).properties(
            width=800,
            height=400,
            title=f"Historical Prices for {ticker_input.upper()}"
        ).interactive()
        
        st.subheader("Historical Data")
        st.altair_chart(hist_chart, use_container_width=True)
        
        # --------------------------
        #  2) SIMULATIONS CHART
        # --------------------------
        sim_chart = alt.Chart(sims_df).mark_line(opacity=0.6).encode(
            x=alt.X('Date:T', title='Date'),
            y=alt.Y('Price:Q', title='Price'),
            color=alt.Color('Simulation:N', legend=alt.Legend(title="Simulations")),
            tooltip=['Date:T', 'Price:Q', 'Simulation']
        ).properties(
            width=800,
            height=400,
            title="Simulated Trajectories (Monte Carlo)"
        ).interactive()
        
        st.subheader("Monte Carlo Simulations")
        st.altair_chart(sim_chart, use_container_width=True)


if __name__ == "__main__":
    main()

st.write("\n\n\n\n\n")




st.subheader("Source code")
code = """
class Stock():
        def __init__(self, ticker) -> None:
            self.ticker = ticker
        
        def calculate_stock_prices(self, start= "2020-01-01", end= "2024-01-01"):
            import yfinance as yf
            import pandas as pd
            self.returns = pd.DataFrame(yf.download(tickers =  self.ticker, start=start, end = end)["Close"].reset_index())

            self.returns = self.returns.tail(200).reset_index(drop=True)

            return self.returns
        
        def calculate_log_returns(self):
            import numpy as np
            self.returns['log_ret'] = np.log(self.returns["AAPL"]) - np.log(self.returns["AAPL"].shift(1))   
            return self.returns['log_ret']
        
        def montecarlo(self, number_sims, duration):
            import numpy as np
            import matplotlib.pyplot as plt
            import pandas as pd

            plt.figure(figsize=(10,6))
            plt.plot(self.returns["Date"], self.returns["AAPL"], label="Historique")

            mean_returns, std_returns = self.returns["log_ret"].mean(), self.returns["log_ret"].std()
            
            initial = self.returns["AAPL"].iloc[-1]
            last_date = self.returns["Date"].iloc[-1]

            for i in range(1,number_sims + 1):
                sim_rets = np.random.normal(mean_returns, std_returns, duration)
                sim_prices = initial * (sim_rets + 1).cumprod()

                sim_dates = pd.date_range(start= last_date + pd.Timedelta(days=1),
                                    periods=duration, freq='D')

                plt.plot(sim_dates, sim_prices, alpha=0.6)

            return sim_rets
"""
st.code(code, language="python")
