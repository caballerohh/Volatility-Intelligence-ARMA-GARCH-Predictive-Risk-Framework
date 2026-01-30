# Volatility-Forecasting-Risk-Modeling-ARMA-GARCH-Framework

This repository implements an advanced econometric framework using ARMA-GARCH models to capture conditional heteroskedasticity and improve the accuracy of Value at Risk (VaR) estimations. The project moves beyond static historical analysis to provide a dynamic view of risk across a diversified asset universe.

Objective: To develop a robust predictive engine for financial volatility using non-normal distributions (t-Student, GED) to optimize risk management and capital allocation.

Extended Version
This project focuses on the quantitative modeling of financial time series with a weekly frequency (January 2015 to November 2025). By applying ARMA(0,0)-GARCH(1,1) models, the analysis successfully filters returns to identify "Volatility Clustering" and leverage effects, ensuring that risk metrics like VaR 95% are statistically validated through rigorous backtesting.

Key Objectives of the Analysis
•	Time Series Diagnosis: Implementation of Augmented Dickey-Fuller (ADF) tests for stationarity and Ljung-Box tests to identify autocorrelation in returns and squared residuals.
•	Conditional Volatility Modeling: Calibration of GARCH(1,1) models to capture the time-varying nature of risk, identifying periods of high-stress persistence ($\alpha + \beta$ near 1).
•	Distribution Optimization: Comparative analysis between Normal, t-Student, and Generalized Error Distribution (GED) using the Akaike Information Criterion (AIC) to manage the leptokurtosis (fat tails) typical of financial data.
•	Model Validation (Backtesting): Application of Kupiec (Unconditional Coverage) and Christoffersen (Conditional Coverage) tests to evaluate the reliability of VaR thresholds and minimize model risk.

Assets Analyzed
The study covers a broad spectrum of market dynamics:
•	Growth & Tech: Apple (AAPL), Amazon (AMZN).
•	Financial & Systemic: JPMorgan (JPM), Goldman Sachs (GS).
•	Fixed Income & Safe Haven: Vanguard Total Bond Market (BND).
•	Market Benchmarks: S&P 500 (SPY), Dow Jones (DIA).
•	Synthetic Portfolio: A strategically diversified allocation designed for efficiency.

Key Portfolio Results
•	Risk-Adjusted Performance: The portfolio achieved a Sharpe Ratio of 0.70, demonstrating superior efficiency compared to individual systemic banking assets.
•	Volatility Profile: Annualized portfolio volatility was contained at 0.19, with a Beta of 1.01, reflecting a neutral and balanced sensitivity to the broader market.
•	VaR Modeling: The weekly VaR 95% was established at -4.12%, providing a statistically sound threshold for downside protection.
•	Statistical Superiority: The ARMA-GARCH with t-Student distribution proved to be the optimal model (AIC: 2592) for the portfolio, successfully capturing extreme market events.

Code Structure (Pipeline Stages)
•	Stage 1: Diagnosis: Automated scripts to detect stationarity and ARCH effects (Conditional Heteroskedasticity) in raw price series.
•	Stage 2: Parameter Estimation: Iterative calibration of model parameters ($\omega, \alpha, \beta$) across multiple statistical distributions.
•	Stage 3: Risk Forecasting: Generation of dynamic VaR envelopes and tracking of "violations" (failures) during the backtesting period.
•	Stage 4: Residual Analysis: Validation of model fit through ACF/PACF plots of standardized residuals to ensure no remaining autocorrelation.
Technologies/Concepts Used
•	Financial Econometrics: ARMA-GARCH Modeling, Heteroskedasticity, Volatility Clustering.
•	Quantitative Risk Management: Value at Risk (VaR), Backtesting (Kupiec/Christoffersen Tests).
•	Statistical Distributions: t-Student, GED (Generalized Error Distribution), Gaussian Models.
•	Technical Stack: Python (Pandas, NumPy, Arch Library), Time Series Analysis, Hypothesis Testing.
