
"""This script fits Merton jump diffusion models to multiple commodity price indices
and computes the Pearson correlation matrix of their log returns.

The commodity indices from combined_price_indices.csv are hard coded for simplicity.

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ---------- Merton jump diffusion negative log-likelihood ----------
def merton_neg_ll(params, r, dt=1/12, n_max=5):
    mu, sigma, lam, mJ, sJ = params
    if sigma <= 1e-8 or lam < 0 or sJ < 0:
        return 1e12

    # Jump correction term
    kappa = np.exp(mJ + 0.5*sJ**2) - 1
    base_mean = (mu - 0.5*sigma**2 - lam*kappa) * dt
    base_var  = sigma**2 * dt

    # Poisson weights
    pois = [np.exp(-lam*dt)]
    for n in range(1, n_max+1):
        pois.append(pois[-1] * (lam*dt)/n)
    pois = np.array(pois)/np.sum(pois)

    ll = 0
    for rt in r:
        means = base_mean + np.arange(n_max+1)*mJ
        vars_ = base_var  + np.arange(n_max+1)*(sJ**2)

        log_comp = np.log(pois) - 0.5*(np.log(2*np.pi*vars_) + (rt - means)**2/vars_)
        m = np.max(log_comp)
        ll += m + np.log(np.sum(np.exp(log_comp - m)))
    return -ll

# ---------- Fit parameters ----------
def fit_merton(r):
    r_mean, r_var = np.mean(r), np.var(r, ddof=1)
    mu0    = r_mean/(1/12) + 0.5*(r_var/(1/12))
    sigma0 = np.sqrt(max(r_var/(1/12),1e-6))
    x0 = np.array([mu0, sigma0, 0.2, 0.0, 0.05])
    bnds=[(-5,5),(1e-6,5),(0,10),(-1,1),(1e-6,2)]

    res = minimize(merton_neg_ll, x0, args=(r,), method="L-BFGS-B", bounds=bnds)
    params = res.x
    LL = -merton_neg_ll(params, r)
    AIC = 2*5 - 2*LL
    BIC = 5*np.log(len(r)) - 2*LL
    return params, LL, AIC, BIC

# ---------- Merton PDF for plotting ----------
def merton_pdf(x, params, dt=1/12, n_max=5):
    mu, sigma, lam, mJ, sJ = params
    kappa = np.exp(mJ + 0.5*sJ**2) - 1
    base_mean = (mu - 0.5*sigma**2 - lam*kappa) * dt
    base_var  = sigma**2 * dt

    pois = [np.exp(-lam*dt)]
    for n in range(1, n_max+1):
        pois.append(pois[-1]*(lam*dt)/n)
    pois = np.array(pois)/np.sum(pois)

    pdf = np.zeros_like(x)
    for n in range(n_max+1):
        mean_n = base_mean + n*mJ
        var_n  = base_var  + n*(sJ**2)
        pdf += pois[n] * (1/np.sqrt(2*np.pi*var_n)) * np.exp(-(x-mean_n)**2/(2*var_n))
    return pdf

# ---------- MAIN ----------
def main():
    df = pd.read_csv("combined_price_indices.csv", sep=";")
    df = df.sort_values("observation_date")
    cols = ["Carbon","Steel","Copper"]

    returns = {}
    results = {}

    print("\n=== Fitting Merton Jump Diffusion ===")

    # Fit parameters for each commodity
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna().values
        r = np.diff(np.log(s))
        returns[c] = r

        params, LL, AIC, BIC = fit_merton(r)
        results[c] = (params, LL, AIC, BIC)

        mu, sigma, lam, mJ, sJ = params

        print(f"\n{c}:")
        print(f"  drift (mu):          {mu:.6f}")
        print(f"  volatility (sigma):  {sigma:.6f}")
        print(f"  jump intensity λ:    {lam:.6f}")
        print(f"  jump size (mu_J):    {mJ:.6f}")
        print(f"  jump vol (sigma_J):  {sJ:.6f}")
        print(f"  log-likelihood:      {LL:.3f}")
        print(f"  AIC:                 {AIC:.3f}")
        print(f"  BIC:                 {BIC:.3f}")

        # Plot #1: empirical histogram + MJD PDF
        xgrid = np.linspace(r.min(), r.max(), 400)
        pdf = merton_pdf(xgrid, params)
        plt.figure(figsize=(7,4))
        plt.hist(r, bins=40, density=True, alpha=0.6, color="steelblue")
        plt.plot(xgrid, pdf, "r-", lw=2)
        plt.title(f"{c} – JD Fit")
        plt.xlabel("Log return")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.savefig(f"{c}_JD_fit.png")
        plt.close()

    # ---------- Pearson Correlation Matrix ----------
    min_len = min(len(returns[c]) for c in cols)
    mat = np.vstack([returns[c][:min_len] for c in cols]).T
    corr = pd.DataFrame(mat, columns=cols).corr()

    print("\n=== Pearson Correlation Matrix (log returns) ===")
    print(corr)

if __name__ == "__main__":
    main()
