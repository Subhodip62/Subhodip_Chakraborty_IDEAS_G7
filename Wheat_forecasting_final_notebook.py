import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# Load Data
df = pd.read_csv("merged_wheat_reservoir.csv", parse_dates=['temperature_recorded_date']) #Change the dataset name
df.rename(columns={'temperature_recorded_date': 'date'}, inplace=True)

# Select numeric columns & convert 
exog_vars = [
    'state_temperature_max_val',
    'state_temperature_min_val',
    'state_rainfall_val',
    'FRL',
    'Live Cap FRL',
    'Level',
    'Current Live Storage'
]

# Convert to numeric
for col in exog_vars + ['yield']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with NaNs in key columns
df = df.dropna(subset=['yield'] + exog_vars)

# Rename for safe access
df.rename(columns={
    'Live Cap FRL': 'Live_CAP_FRL',
    'Current Live Storage': 'Storage'
}, inplace=True)

exog_vars = [
    'state_temperature_max_val',
    'state_temperature_min_val',
    'state_rainfall_val',
    'FRL',
    'Live_CAP_FRL',
    'Level',
    'Storage'
]

# States to process
states_to_process = df['state_name'].unique()

# Create a DataFrame to collect forecasts
all_forecasts = []

for state in states_to_process:
    
    print(f"Processing State: {state}")
   

    try:
        # Filter state data
        df_state = df[df['state_name'] == state].copy()
        df_state['date'] = pd.to_datetime(df_state['date'], errors='coerce')
        df_state = df_state.dropna(subset=['date'])

        # Ensure datetime index
        df_state = df_state.set_index('date')

        # Keep only numeric columns for resampling
        df_state_numeric = df_state.select_dtypes(include=['number'])

        # Step 1: Resample to monthly means
        df_monthly = df_state_numeric.resample('M').mean()

        # Step 2: Sum the monthly means across each year
        df_yearly = df_monthly.resample('Y').sum().dropna()

        # Keep only target + exogenous variables
        df_yearly = df_yearly[['yield'] + exog_vars].dropna()

        # Ensure enough data
        if len(df_yearly) < 5:
            print(f"⚠️ Not enough data for ARIMAX in {state} ({len(df_yearly)} years). Skipping...")
            continue

        # Add Year column for later use
        df_yearly['Year'] = df_yearly.index.year
    
        # Plot Yield vs Time
        plt.figure(figsize=(12, 5))
        plt.plot(df_yearly.index.year, df_yearly['yield'], label='Annual Yield', marker='o', color='blue')
        plt.title(f"{state} - Annual Wheat Yield")
        plt.xlabel("Year")
        plt.ylabel("Yield")
        plt.grid(True, alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Decomposition
        if len(df_yearly) >= 5:  # At least 5 years of annual data
            for model in ['additive']:
                decomposition = seasonal_decompose(df_yearly['yield'], model=model, period=1)
                fig = decomposition.plot()
                fig.set_size_inches(10, 8)

                # Rename Y-axis label for the observed series to include "Original"
                fig.axes[0].set_ylabel("Original")
                fig.suptitle(f"{state} - {model.capitalize()} Decomposition (Annual Data)", fontsize=14)
                plt.tight_layout()
                plt.show()
        # Stationarity Tests (ADF + KPSS) + ACF/PACF
        from statsmodels.tsa.stattools import adfuller, kpss
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

        series = df_yearly['yield'].dropna().copy()

        # Step 1: ADF and KPSS Tests
        adf_result = adfuller(series)
        print(f"ADF Statistic: {adf_result[0]:.4f}, p-value: {adf_result[1]:.4f}")

        try:
            kpss_result = kpss(series, regression='c', nlags='auto')
            print(f"KPSS Statistic: {kpss_result[0]:.4f}, p-value: {kpss_result[1]:.4f}")
        except Exception as e:
            print(f" KPSS test failed: {e}")
            kpss_result = None

        # Step 2: Plot ACF and PACF for Original
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        plot_acf(series, ax=axes[0], lags=min(20, len(series)//2), title=f"{state} - Original Series ACF")
        plot_pacf(series, ax=axes[1], lags=min(20, len(series)//2), title=f"{state} - Original Series PACF")
        plt.tight_layout()
        plt.show()

        # Step 3: Check Stationarity
        is_adf_stationary = adf_result[1] < 0.05
        is_kpss_stationary = kpss_result is not None and kpss_result[1] > 0.05

        if not (is_adf_stationary and is_kpss_stationary):
            print(" Series non-stationary. Applying first differencing...")
            series_diff = series.diff().dropna().reset_index(drop=True)

            # Step 4: ADF + KPSS again
            adf_result_diff = adfuller(series_diff)
            print(f"After differencing → ADF Statistic: {adf_result_diff[0]:.4f}, p-value: {adf_result_diff[1]:.4f}")
            try:
                kpss_result_diff = kpss(series_diff, regression='c', nlags='auto')
                print(f"After differencing → KPSS Statistic: {kpss_result_diff[0]:.4f}, p-value: {kpss_result_diff[1]:.4f}")
            except:
                print("KPSS failed on differenced series.")


            # Step 5: Plot differenced series
            plt.figure(figsize=(12, 5))
            plt.plot(series.index[1:], series_diff, color='orange', label='Differenced')
            plt.title(f"{state} - Yield After First Differencing")
            plt.xlabel("Year")
            plt.ylabel("Yield (Differenced)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

            for model in ['additive']:
                decomposition = seasonal_decompose(series_diff, model=model, period=1)
                fig = decomposition.plot()
                fig.set_size_inches(10, 8)

                # Rename Y-axis label for the observed series to include "Original"
                fig.axes[0].set_ylabel("Original")
                fig.suptitle(f"{state} - {model.capitalize()} Decomposition (Annual Differenced Data)", color = 'orange',fontsize=14)
                plt.tight_layout()
                plt.show()

            # Step 6: ACF + PACF after differencing
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            plot_acf(series_diff, ax=axes[0], lags=min(20, len(series_diff)//2), title=f"{state} - Differenced Series ACF")
            plot_pacf(series_diff, ax=axes[1], lags=min(20, len(series_diff)//2), title=f"{state} - Differenced Series PACF")
            plt.tight_layout()
            plt.show()
        else:
            print("✅ Series is stationary. Proceeding to forecasting.")
             # Step 3: Check Stationarity again
        is_adf_diff_stationary = adf_result_diff[1] < 0.05
        is_kpss_diff_stationary = kpss_result_diff is not None and kpss_result_diff[1] > 0.05

        if not (is_adf_diff_stationary and is_kpss_diff_stationary):
            print(" Series non-stationary. Applying second differencing...")
            series_diff_2 = series_diff.diff().dropna().reset_index(drop=True)

            # Step 4: ADF + KPSS again
            adf_result_diff_2= adfuller(series_diff_2)
            print(f"After differencing → ADF Statistic: {adf_result_diff_2[0]:.4f}, p-value: {adf_result_diff_2[1]:.4f}")
            try:
                kpss_result_diff_2 = kpss(series_diff_2, regression='c', nlags='auto')
                print(f"After differencing → KPSS Statistic: {kpss_result_diff_2[0]:.4f}, p-value: {kpss_result_diff_2[1]:.4f}")
            except:
                print("KPSS failed on differenced series.")


            # Step 5: Plot differenced series
            plt.figure(figsize=(12, 5))
            plt.plot(series_diff.index[1:], series_diff_2, color='black', label='Differenced')
            plt.title(f"{state} - Yield After Second Differencing")
            plt.xlabel("Year")
            plt.ylabel("Yield (Differenced)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

            for model in ['additive']:
                decomposition = seasonal_decompose(series_diff_2, model=model, period=1)
                fig = decomposition.plot()
                fig.set_size_inches(10, 8)

                # Rename Y-axis label for the observed series to include "Original"
                fig.axes[0].set_ylabel("Original")
                fig.suptitle(f"{state} - {model.capitalize()} Decomposition (Annual Differenced Data)",fontsize=14)
                plt.tight_layout()
                plt.show()

            # Step 6: ACF + PACF after differencing
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            plot_acf(series_diff, ax=axes[0], lags=min(20, len(series_diff_2)//2), title=f"{state} - Differenced Series ACF")
            plot_pacf(series_diff, ax=axes[1], lags=min(20, len(series_diff_2)//2), title=f"{state} - Differenced Series PACF")
            plt.tight_layout()
            plt.show()
    
        else:
            print(" Series is stationary. Proceeding to forecasting.")

        # Train/Test Split 
        train = df_yearly['2000':'2021']
        test = df_yearly['2022':]

        exog_train = train[exog_vars]
        exog_test = test[exog_vars]

        y_train = train['yield']
        y_test = test['yield']

        # ARIMAX Model Selection (p,d,q) 
        import itertools
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        best_aic = float('inf')
        best_order = None
        best_model = None

        for p, d, q in itertools.product([0, 1],[1,2],[0,1]):
            order = (p, d, q)
            try:
                model = SARIMAX(y_train, order=order, exog=exog_train, enforce_stationarity=False, enforce_invertibility=False)
                model_fit = model.fit(disp=False)
                if model_fit.aic < best_aic:
                    best_aic = model_fit.aic
                    best_order = order
                    best_model = model_fit
                    print(f"Checked ARIMAX{order}, AIC={model_fit.aic:.2f}")
            except Exception as e:
                print(f" ARIMAX{order} failed: {type(e).__name__} - {e}")
                continue
    


        print(f"\n Best ARIMAX order: {best_order} with AIC={best_aic:.2f}")

        # Forecast Future (2023–2025)
        forecast_years = [2023, 2024, 2025]
        forecast_steps = len(forecast_years)

        # Use last available exogenous row for projection
        last_exog = df_yearly[exog_vars].iloc[-1:].values
        exog_future = np.tile(last_exog, (forecast_steps, 1))

        # Generate forecast
        forecast_values = best_model.forecast(steps=forecast_steps, exog=exog_future)
        forecast_index = pd.date_range(start='2023', periods=forecast_steps, freq='Y')

        #  Evaluation on test
        y_pred_test = best_model.forecast(steps=len(test), exog=exog_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        mse = mean_squared_error(y_test, y_pred_test)
        print(f"MAE: {mae:.3f}, MSE: {mse:.3f}")

        # Ensure proper datetime index
        df_yearly.index = pd.to_datetime(df_yearly.index, format='%Y')
        forecast_index = pd.date_range(start=df_yearly.index[-1] + pd.offsets.YearBegin(),
                                        periods=len(forecast_values), freq='YS')

        # Plot
        plt.figure(figsize=(12, 5))
        plt.plot(df_yearly.index, df_yearly['yield'], label='Observed', color='steelblue', marker='o')
        plt.plot([df_yearly.index[-1]] + list(forecast_index),
                [df_yearly['yield'].iloc[-1]] + list(forecast_values),
                label='Forecast (2023–2025)', color='orange', marker='o')

        plt.title(f"{state} - ARIMAX Forecast (2023–2025)")
        plt.xlabel("Year")
        plt.ylabel("Yield")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f" Error processing {state}: {e}")

