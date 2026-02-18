"""
NBA Player Performance Prediction System

This script implements a machine learning pipeline to predict NBA players'
points per 36 minutes for the next season using historical performance data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

class NBAPlayerPredictor:
    def __init__(self, data_path):
        """Initialize the predictor with the path to the NBA data."""
        self.data_path = data_path
        self.data = None
        self.models = {}
        self.features = []
        self.target = 'pts_per_36_min_next'
        self.scaler = StandardScaler()
        
    def load_and_preprocess(self):
        """Load and preprocess the NBA player data."""
        # Load the data
        self.data = pd.read_csv(self.data_path)
        
        # Sort by player_id and season for time series operations
        self.data = self.data.sort_values(['player_id', 'season'])
        
        # Create target variable (next season's points per 36 minutes)
        self.data['pts_per_36_min_next'] = self.data.groupby('player_id')['pts_per_36_min'].shift(-1)
        
        # Remove rows where target is NaN (last season for each player)
        self.data = self.data.dropna(subset=['pts_per_36_min_next'])
        
        return self.data
    
    def create_features(self):
        """Create features for the prediction model."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_and_preprocess() first.")
            
        # List of numeric features to create lags and rolling stats for
        numeric_cols = ['pts_per_36_min', 'fg_per_36_min', 'fga_per_36_min', 'fg_percent', 
                       'x3p_per_36_min', 'x3pa_per_36_min', 'x3p_percent', 
                       'ft_per_36_min', 'fta_per_36_min', 'ft_percent', 
                       'trb_per_36_min', 'ast_per_36_min', 'stl_per_36_min', 
                       'blk_per_36_min', 'tov_per_36_min', 'pf_per_36_min', 'g', 'mp']
        
        # Create lagged features (previous season's stats)
        for col in numeric_cols:
            self.data[f'{col}_lag1'] = self.data.groupby('player_id')[col].shift(1)
            
        # Calculate year-over-year changes
        for col in numeric_cols:
            self.data[f'{col}_yoy'] = self.data.groupby('player_id')[col].pct_change()
        
        # Create rolling statistics (2-season window)
        for col in numeric_cols:
            self.data[f'{col}_rollmean'] = self.data.groupby('player_id')[col].transform(
                lambda x: x.rolling(window=2, min_periods=1).mean()
            )
            self.data[f'{col}_rollstd'] = self.data.groupby('player_id')[col].transform(
                lambda x: x.rolling(window=2, min_periods=1).std()
            )
        
        # Define feature columns
        self.features = [
            'age', 'g', 'mp',
            *[f'{col}_lag1' for col in numeric_cols],
            *[f'{col}_yoy' for col in numeric_cols if col not in ['g', 'mp']],
            *[f'{col}_rollmean' for col in numeric_cols],
            *[f'{col}_rollstd' for col in numeric_cols]
        ]
        
        # Add position dummies
        if 'pos' in self.data.columns:
            pos_dummies = pd.get_dummies(self.data['pos'], prefix='pos', drop_first=True)
            self.data = pd.concat([self.data, pos_dummies], axis=1)
            self.features.extend(pos_dummies.columns.tolist())
        
        # Drop rows with missing values in features
        self.data = self.data.dropna(subset=self.features + [self.target])
        
        # Replace infinite values with NaN and then drop them
        self.data = self.data.replace([np.inf, -np.inf], np.nan)
        self.data = self.data.dropna(subset=self.features + [self.target])
        
        return self.data
    
    def prepare_train_test_split(self, test_size=0.2):
        """Split data into training and testing sets using time-based split."""
        if not self.features:
            raise ValueError("No features available. Call create_features() first.")
        
        # Sort by season to ensure time-based split
        self.data = self.data.sort_values('season')
        
        # Split data by time
        train_size = int(len(self.data) * (1 - test_size))
        X_train = self.data[self.features].iloc[:train_size]
        X_test = self.data[self.features].iloc[train_size:]
        y_train = self.data[self.target].iloc[:train_size]
        y_test = self.data[self.target].iloc[train_size:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """Train multiple regression models."""
        # Initialize models
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, random_state=42, eval_metric='rmse')
        }
        
        # Train each model
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
        
        return self.models
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate models on test data and return metrics."""
        results = []
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            results.append({
                'Model': name,
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2
            })
            
            print(f"\n{name} Performance:")
            print(f"MAE: {mae:.2f}")
            print(f"RMSE: {rmse:.2f}")
            print(f"R²: {r2:.4f}")
        
        return pd.DataFrame(results)
    
    def plot_feature_importance(self, model_name='Random Forest', top_n=15):
        """Plot feature importance for the specified model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models
            importances = np.abs(model.coef_)
        else:
            raise ValueError("Model doesn't support feature importance analysis.")
        
        # Create a DataFrame for visualization
        feature_importance = pd.DataFrame({
            'feature': self.features,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title(f'Top {top_n} Important Features - {model_name}')
        plt.tight_layout()
        plt.show()
        
        return feature_importance
    
    def predict_player_season(self, player_data):
        """
        Predict next season's points per 36 minutes for a player.
        
        Args:
            player_data (pd.DataFrame): DataFrame containing the player's current season stats
                                       with the same features used for training.
        
        Returns:
            dict: Predictions from all models with confidence intervals.
        """
        if not self.models:
            raise ValueError("No models trained. Call train_models() first.")
        
        # Ensure all required features are present
        missing_features = set(self.features) - set(player_data.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Select and scale features
        X = player_data[self.features]
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = {}
        for name, model in self.models.items():
            pred = model.predict(X_scaled)
            predictions[name] = {
                'prediction': pred[0],
                'current_pts_per_36': player_data['pts_per_36_min'].values[0] if 'pts_per_36_min' in player_data else None
            }
        
        return predictions

def main():
    # Initialize the predictor
    predictor = NBAPlayerPredictor(r'C:\Users\VICTUS\OneDrive\Desktop\SEM V\ML\NBA_36.csv')
    
    try:
        # Load and preprocess data
        print("Loading and preprocessing data...")
        data = predictor.load_and_preprocess()
        
        # Create features
        print("\nCreating features...")
        data_with_features = predictor.create_features()
        
        # Prepare train/test split
        print("\nPreparing train/test split...")
        X_train, X_test, y_train, y_test = predictor.prepare_train_test_split()
        
        # Train models
        print("\nTraining models...")
        models = predictor.train_models(X_train, y_train)
        
        # Evaluate models
        print("\nEvaluating models...")
        results = predictor.evaluate_models(X_test, y_test)
        
        # Show model comparison
        print("\nModel Comparison:")
        print(results.to_string(index=False))
        
        # Plot feature importance for the best model
        print("\nPlotting feature importance...")
        best_model = results.loc[results['R2'].idxmax(), 'Model']
        feature_importance = predictor.plot_feature_importance(model_name=best_model)
        
        print("\nPrediction pipeline completed successfully!")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    main()
