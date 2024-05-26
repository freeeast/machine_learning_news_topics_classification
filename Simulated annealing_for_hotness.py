import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import dual_annealing

hotness_metrics_df = pd.read_csv('hotness_metrics_clustered1.csv')

features = [
    'max_hotness', 'gt_mean_duration', 'hotness_chg_rate', 'hotness_area',
    'num_peaks', 'avg_peak_interval', 'rise_rate', 'fall_rate', 'std_dev', 
    'autocorr', 'seasonal_std'
]

clusters_to_model = [6, 18, 4, 1, 12]

metrics = ['MAE', 'MSE', 'RMSE', 'R2']

results_df = pd.DataFrame(columns=['Cluster'] + metrics)

for cluster in clusters_to_model:
    cluster_data = hotness_metrics_df[hotness_metrics_df['cluster'] == cluster]
    
    X = cluster_data[features]
    y = cluster_data['max_hotness'] 
    
    if len(cluster_data) >= 5: 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X, None, y, None 
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    def objective(weights):
        model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=1000, random_state=42)
        model.coefs_ = [weights[:64*11].reshape(11, 64), weights[64*11:64*11+64*32].reshape(64, 32)]
        model.intercepts_ = [weights[64*11+64*32:64*11+64*32+64], weights[64*11+64*32+64:]]
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_train_scaled)
        mse = mean_squared_error(y_train, y_pred)
        return mse
    
    bounds = [(-1, 1)] * (64*11 + 64*32 + 64 + 32)
    result = dual_annealing(objective, bounds)
    best_weights = result.x
    
    model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=1000, random_state=42)
    model.coefs_ = [best_weights[:64*11].reshape(11, 64), best_weights[64*11:64*11+64*32].reshape(64, 32)]
    model.intercepts_ = [best_weights[64*11+64*32:64*11+64*32+64], best_weights[64*11+64*32+64:]]
    model.fit(X_train_scaled, y_train)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
    else:
        mae, mse, rmse, r2 = None, None, None, None
    
    results_df = pd.concat([results_df, pd.DataFrame({'Cluster': [cluster], 'MAE': [mae], 'MSE': [mse], 'RMSE': [rmse], 'R2': [r2]})], ignore_index=True)

print("Result:")
print(results_df)