India is the 2nd largest producer of wheat in the world and it is the second most produced staple food crop of the country. 
The daily production of wheat in 14 different states of India, during 2000–2022 are used here to predict the same for 2023-2025. 
The dataset, comprising of daily data on wheat yield, temperature, rainfall, reservoir levels and some other variables, was first converted into monthly data by taking the arithmetic means of the values available for each month and then converted into annual data by adding those means. 
Further, it was decomposed to check the presence of trend and seasonality. 
Before fitting the model, it was necessary to know whether the data was stationary or not. 
In this regard, statistical tests for stationarity were applied, and for the states with non-stationary data, differencing were applied to make them stationary. 
Due to the presence of exogenous variables, such as temperature in the state, amount of rainfall, capacity of reservoir, and amount of water available, in the dataset, ARIMAX model was used. 
Also, the validity and readability of the ARIMAX models were further enhanced by various model evaluation techniques such as residual analysis and model selection criteria like Akaike Information Criterion (AIC). 
Best predictions were found for the states Rajasthan, Andhra Pradesh and Madhya Pradesh where ARIMAX (0,2,1), ARIMAX (0,1,1) and ARIMAX (0,2,1) were fitted respectively.
