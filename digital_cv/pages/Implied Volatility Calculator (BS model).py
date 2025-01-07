import streamlit as st
import numpy as np
from scipy.stats import norm

# Fonctions de calcul du prix de l'option et de la volatilité implicite
def pricing_option(S, K, r, q, sigma, T, call=True):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if call:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_vol(S, K, r, q, T, price, call=True):
    tol = 1e-5
    max_iter = 300
    sigma = 0.5
    for i in range(max_iter):
        price_est = pricing_option(S, K, r, q, sigma, T, call)
        vega = S * np.sqrt(T) * norm.pdf((np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T)))
        diff = price_est - price
        if abs(diff) < tol:
            return sigma
        sigma -= diff / vega
    return sigma

#Implied vol using bissection method
def implied_vol_bissection(S, K, r, q, T, price, call=True):
    tol = 1e-5
    max_iter = 300
    sigma_low = 0.0001
    sigma_high = 5
    for i in range(max_iter):
        sigma = (sigma_low + sigma_high) / 2
        price_est = pricing_option(S, K, r, q, sigma, T, call)
        if price_est > price:
            sigma_high = sigma
        else:
            sigma_low = sigma
        if abs(price_est - price) < tol:
            return sigma
    return sigma

#Implied vol using secant method
def implied_vol_secant(S, K, r, q, T, price, call=True):
    tol = 1e-5
    max_iter = 300
    sigma_0 = 0.0001
    sigma_1 = 0.5
    for i in range(max_iter):
        price_0 = pricing_option(S, K, r, q, sigma_0, T, call)
        price_1 = pricing_option(S, K, r, q, sigma_1, T, call)
        sigma = sigma_1 - (price_1 - price) * (sigma_1 - sigma_0) / (price_1 - price_0)
        if abs(sigma - sigma_1) < tol:
            return sigma
        sigma_0 = sigma_1
        sigma_1 = sigma



# Interface Streamlit
st.title("Implied Volatility Calculator using Different Methods")

st.write("""This calculator allows you to compute the price of a European option and its implied volatility. 
            The implied volatility is estimated by using the Newton-Raphson method. 
            The price of the option is calculated using the Black-Scholes formula.""")

# Utilisation de colonnes pour organiser les entrées
col1, col2, col3 = st.columns(3)

with col1:
    S = st.number_input("Spot Price (S)", value=100.0)
    K = st.number_input("Strike Price (K)", value=100.0)

with col2:
    r = st.number_input("Risk free rate (r)", value=0.05)
    q = st.number_input("Dividen rate (q)", value=0.0)

with col3:
    sigma = st.number_input("Volatility (sigma)", value=0.2)
    T = st.number_input("Remaining time before strike (T, in years)", value=1.0)

call = st.radio("Option Type", ("Call", "Put")) == "Call"

# Bouton pour déclencher le calcul
if st.button("Calculate"):
    cola,colb,colc = st.columns(3)
    # Option price
    option_price = pricing_option(S, K, r, q, sigma, T, call)
    st.write(f"Option price : {option_price:.2f}")

    with st.container():
        st.subheader("Implied Volatility Calculation Results")

        with st.expander("Newton-Raphson Method"):
            implied_volatility = implied_vol(S, K, r, q, T, option_price, call)
            st.write("The Newton-Raphson method is used to calculate the implied volatility by iteratively solving the Black-Scholes equation.")
            st.write(f"Implied Volatility : {implied_volatility}")
        with st.expander("Bissection Method"):
            implied_volatility_bissection = implied_vol_bissection(S, K, r, q, T, option_price, call)
            st.write('The bissection method is used to calculate the implied volatility by iteratively narrowing the interval where the solution lies.')
            st.write(f"Implied Volatility : {implied_volatility_bissection}")
        with st.expander("Secant Method"):
            implied_volatility_secant = implied_vol_secant(S, K, r, q, T, option_price, call)
            st.write("The secant method is used to calculate the implied volatility by approximating the derivative of the Black-Scholes equation.")
            st.write(f"Implied Volatility : {implied_volatility_secant}") 

st.subheader("Source code")
code = """
import numpy as np
from scipy.stats import norm

# Function for calculating the option price
def pricing_option(S, K, r, q, sigma, T, call=True):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if call:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Implied vol using Newton-Raphson method
def implied_vol(S, K, r, q, T, price, call=True):
    tol = 1e-5
    max_iter = 300
    sigma = 0.5
    for i in range(max_iter):
        price_est = pricing_option(S, K, r, q, sigma, T, call)
        vega = S * np.sqrt(T) * norm.pdf((np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T)))
        diff = price_est - price
        if abs(diff) < tol:
            return sigma
        sigma -= diff / vega
    return sigma

#Implied vol using bissection method
def implied_vol_bissection(S, K, r, q, T, price, call=True):
    tol = 1e-5
    max_iter = 300
    sigma_low = 0.0001
    sigma_high = 5
    for i in range(max_iter):
        sigma = (sigma_low + sigma_high) / 2
        price_est = pricing_option(S, K, r, q, sigma, T, call)
        if price_est > price:
            sigma_high = sigma
        else:
            sigma_low = sigma
        if abs(price_est - price) < tol:
            return sigma
    return sigma

#Implied vol using secant method
def implied_vol_secant(S, K, r, q, T, price, call=True):
    tol = 1e-5
    max_iter = 300
    sigma_0 = 0.0001
    sigma_1 = 0.5
    for i in range(max_iter):
        price_0 = pricing_option(S, K, r, q, sigma_0, T, call)
        price_1 = pricing_option(S, K, r, q, sigma_1, T, call)
        sigma = sigma_1 - (price_1 - price) * (sigma_1 - sigma_0) / (price_1 - price_0)
        if abs(sigma - sigma_1) < tol:
            return sigma
        sigma_0 = sigma_1
        sigma_1 = sigma
"""
st.code(code, language="python")