import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

def black_scholes(S, X, T, r, sigma, opt):
    d1 = (np.log(S/X) + (r+0.5*sigma**2) * T) / (sigma*np.sqrt(T))
    d2 = (np.log(S/X)+(r-0.5*sigma**2)*T) / (sigma*np.sqrt(T))

    if opt == 'call':
        return S * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)
    elif opt == 'put':
        return X * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid: Use 'call' or 'put'")

# works for a call or a put option
def implied_volatility(S, X, T, r, market_price, opt):
    def obj_fun(sigma):
        if sigma <= 0:
            return np.inf
        theo_price = black_scholes(S, X, T, r, sigma, opt)
        return (theo_price - market_price) ** 2

    init_guess = 0.2
    bounds = [(1e-5, 5)]

    result = minimize(obj_fun, init_guess, bounds=bounds, method='L-BFGS-B')

    if result.success:
        return result.x[0]
    else:
        return np.nan

# test
S = 60
X = 65
r = 0.08
T = 0.25
market_price = 4

iv = implied_volatility(S, X, T, r, market_price, opt='call')
print(f"Implied Volatility: {iv:.4f}")