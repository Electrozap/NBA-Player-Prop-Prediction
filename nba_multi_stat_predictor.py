"""
NBA Player Multi-Stat Prediction System

This script predicts multiple statistics (points, rebounds, assists, steals, blocks)
for NBA players for the next season.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
import argparse
warnings.filterwarnings('ignore')

class NBAMultiStatPredictor:
    def __init__(self, data_path):
        """Initialize the predictor with the path to the NBA data."""
        self.data_path = data_path
        self.data = None
        self.models = {}
        self.features = []
        self.scalers = {}
        
        # Define target statistics to predict
        self.target_stats = {
            'pts_per_36_min': 'Points',
            'trb_per_36_min': 'Rebounds',
            'ast_per_36_min': 'Assists',
            'stl_per_36_min': 'Steals',
            'blk_per_36_min': 'Blocks'
        }
        
    def load_and_preprocess(self):
        """Load and basic preprocess (sort) the NBA player data."""
        self.data = pd.read_csv(self.data_path)
        self.data = self.data.sort_values(['player_id', 'season'])
        return self.data
    
    def create_supervised_from_prev_season(self):
        """Create supervised dataset: X = previous season features, Y = current season targets."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_and_preprocess() first.")
        
        # Feature columns taken from previous season
        base_feature_cols = [
            'age', 'g', 'mp',
            'pts_per_36_min', 'fg_per_36_min', 'fga_per_36_min', 'fg_percent',
            'x3p_per_36_min', 'x3pa_per_36_min', 'x3p_percent',
            'ft_per_36_min', 'fta_per_36_min', 'ft_percent',
            'trb_per_36_min', 'ast_per_36_min', 'stl_per_36_min',
            'blk_per_36_min', 'tov_per_36_min', 'pf_per_36_min'
        ]
        id_cols = ['player_id', 'season']
        optional_cats = ['pos', 'team']
        existing_cats = [c for c in optional_cats if c in self.data.columns]

        # Previous season features: increment season to align with current season targets
        prev_cols = id_cols + base_feature_cols + existing_cats
        df_prev = self.data[prev_cols].copy()
        df_prev['season'] = df_prev['season'] + 1
        df_prev = df_prev.rename(columns={col: f'prev_{col}' for col in base_feature_cols + existing_cats})

        # Current season targets
        target_cols = list(self.target_stats.keys())
        df_curr = self.data[id_cols + ['player'] + target_cols].copy() if 'player' in self.data.columns else self.data[id_cols + target_cols].copy()

        # Merge to create supervised rows: each row uses season-1 features to predict season targets
        supervised = pd.merge(df_curr, df_prev, on=['player_id', 'season'], how='inner')

        # Build feature list from prev_ columns
        self.features = [f'prev_{c}' for c in base_feature_cols]

        # One-hot encode previous season position if available
        if 'prev_pos' in supervised.columns:
            pos_dummies = pd.get_dummies(supervised['prev_pos'], prefix='prev_pos', drop_first=True)
            supervised = pd.concat([supervised, pos_dummies], axis=1)
            self.features.extend(pos_dummies.columns.tolist())

        # Clean NaNs/Infs
        supervised = supervised.replace([np.inf, -np.inf], np.nan)
        supervised = supervised.dropna(subset=self.features + target_cols)

        self.data = supervised.sort_values(['season', 'player_id'])
        return self.data
    
    def train_models(self):
        """Train models with train on seasons < 2025 and test on season == 2025."""
        print("\nTraining models for multiple statistics (train: <2025, test: 2025)...")

        if 'season' not in self.data.columns:
            raise ValueError("'season' column missing from supervised dataset.")

        train_mask = self.data['season'] < 2025
        test_mask = self.data['season'] == 2025

        if test_mask.sum() == 0:
            raise ValueError("No rows found for season 2025 in the dataset.")

        results = []

        for stat, stat_name in self.target_stats.items():
            print(f"\nTraining model for {stat_name}...")

            y_col = stat
            X_train = self.data.loc[train_mask, self.features]
            X_test = self.data.loc[test_mask, self.features]
            y_train = self.data.loc[train_mask, y_col]
            y_test = self.data.loc[test_mask, y_col]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers[stat] = scaler

            model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
            model.fit(X_train_scaled, y_train)
            self.models[stat] = model

            y_pred = model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            results.append({'Statistic': stat_name, 'MAE': mae, 'RMSE': rmse, 'R2': r2})

            print(f"  MAE: {mae:.3f}")
            print(f"  RMSE: {rmse:.3f}")
            print(f"  R²: {r2:.4f}")

        return pd.DataFrame(results)
    
    def predict_player_stats(self, player_data):
        """
        Predict next season's statistics for a player.
        
        Args:
            player_data (pd.DataFrame): DataFrame containing the player's current season stats
        
        Returns:
            dict: Predictions for all statistics
        """
        if not self.models:
            raise ValueError("No models trained. Call train_models() first.")
        
        # Ensure all required features are present
        missing_features = set(self.features) - set(player_data.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        predictions = {}
        
        for stat, stat_name in self.target_stats.items():
            # Select and scale features
            X = player_data[self.features]
            X_scaled = self.scalers[stat].transform(X)
            
            # Make prediction
            pred = self.models[stat].predict(X_scaled)[0]
            current = player_data[stat].values[0] if stat in player_data else None
            
            predictions[stat_name] = {
                'current': current,
                'predicted': pred,
                'change': pred - current if current else None
            }
        
        return predictions

    def evaluate_on_2025(self):
        """Generate player-by-player comparison for 2025 and accuracy plots."""
        if not self.models:
            raise ValueError("Models are not trained.")

        test_df = self.data[self.data['season'] == 2025].copy()
        comparisons = test_df[['player_id', 'season'] + ([ 'player'] if 'player' in test_df.columns else [])].copy()

        for stat, stat_name in self.target_stats.items():
            X = test_df[self.features]
            X_scaled = self.scalers[stat].transform(X)
            y_true = test_df[stat].values
            y_pred = self.models[stat].predict(X_scaled)

            comparisons[f'actual_{stat}'] = y_true
            comparisons[f'pred_{stat}'] = y_pred

            # Parity plot
            plt.figure(figsize=(6, 6))
            plt.scatter(y_true, y_pred, alpha=0.6)
            lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
            plt.plot(lims, lims, 'r--', linewidth=1)
            plt.xlabel(f'Actual {stat_name}')
            plt.ylabel(f'Predicted {stat_name}')
            plt.title(f'2025 Actual vs Predicted - {stat_name}')
            plt.tight_layout()
            plt.savefig(f'accuracy_2025_{stat}.png', dpi=150)
            plt.close()

        # Save comparison CSV
        comparisons.to_csv('player_predictions_2025.csv', index=False)
        return comparisons


def predict_specific_players(predictor, player_names):
    """
    Predict next season's performance for specific players.
    
    Args:
        predictor: Trained NBAMultiStatPredictor instance
        player_names: List of player names to predict
    """
    results = []
    
    for player_name in player_names:
        # Find the player's most recent season data
        player_data = predictor.data[predictor.data['player'].str.contains(player_name, case=False, na=False)]
        
        if player_data.empty:
            print(f"\nPlayer '{player_name}' not found in the dataset.")
            continue
        
        # Get the most recent season for this player
        latest_season = player_data.sort_values('season', ascending=False).iloc[0:1]
        current_season = int(latest_season['season'].values[0])
        predicted_season = current_season + 1
        
        print(f"\n{'='*70}")
        print(f"Player: {latest_season['player'].values[0]}")
        print(f"{'='*70}")
        print(f"Most Recent Season: {current_season}")
        print(f"Predicting For Season: {predicted_season}")
        print(f"Age: {latest_season['age'].values[0]}")
        print(f"Team: {latest_season['team'].values[0]}")
        print(f"Position: {latest_season['pos'].values[0]}")
        print(f"Games Played: {latest_season['g'].values[0]}")
        
        try:
            # Make predictions for all statistics
            predictions = predictor.predict_player_stats(latest_season)
            
            print(f"\nPREDICTIONS FOR {predicted_season} SEASON:")
            print(f"{'Stat':<12} {'Current':<10} {'Predicted':<10} {'Change':<10} {'Trend'}")
            print("-" * 70)
            
            result = {
                'Player': latest_season['player'].values[0],
                'Current_Season': current_season,
                'Predicted_Season': predicted_season,
                'Age': latest_season['age'].values[0],
                'Team': latest_season['team'].values[0],
                'Position': latest_season['pos'].values[0]
            }
            
            for stat_name, pred_data in predictions.items():
                current = pred_data['current']
                predicted = pred_data['predicted']
                change = pred_data['change']
                
                if change is not None:
                    pct_change = (change / current * 100) if current != 0 else 0
                    if change > 0:
                        trend = f"up +{pct_change:.1f}%"
                    elif change < 0:
                        trend = f"down {pct_change:.1f}%"
                    else:
                        trend = "flat 0.0%"
                    
                    print(f"{stat_name:<12} {current:<10.2f} {predicted:<10.2f} {change:<+10.2f} {trend}")
                    
                    result[f'Current_{stat_name}'] = current
                    result[f'Predicted_{stat_name}'] = predicted
                    result[f'Change_{stat_name}'] = change
            
            results.append(result)
        
        except Exception as e:
            print(f"\nError making prediction: {str(e)}")
    
    return pd.DataFrame(results)


def get_user_input():
    """
    Get player names from user input.
    """
    print("\n" + "="*70)
    print("NBA MULTI-STAT PERFORMANCE PREDICTOR".center(70))
    print("="*70)
    print("\nThis tool predicts: Points, Rebounds, Assists, Steals, and Blocks")
    
    # Get player names
    print("\nEnter player names to predict (separated by commas):")
    print("Example: Nikola Jokic, Bam Adebayo, LeBron James")
    try:
        player_input = input("Players: ").strip()
    except EOFError:
        return []
    
    if not player_input:
        print("No players entered. Using default players.")
        players = ['Bam Adebayo', 'Jimmy Butler']
    else:
        players = [name.strip() for name in player_input.split(',')]
    
    print(f"\nWill predict for: {', '.join(players)}")
    
    return players


def main():
    # Parse optional CLI args
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--players', type=str, default='', help='Comma-separated player names to predict for 2025')
    parser.add_argument('--season', type=int, default=2025, help='Season to evaluate/predict (default 2025)')
    try:
        args, _ = parser.parse_known_args()
    except SystemExit:
        class _A: pass
        args = _A(); args.players=''; args.season=2025

    # Initialize the predictor
    print("Initializing NBA Multi-Stat Predictor...")
    predictor = NBAMultiStatPredictor(r'C:\Users\VICTUS\OneDrive\Desktop\SEM V\ML\NBA_36.csv')
    
    try:
        # Load and preprocess data
        print("Loading and preprocessing data...")
        predictor.load_and_preprocess()
        
        # Build supervised dataset using season-1 features and current-season targets
        print("Creating supervised dataset from previous season features...")
        predictor.create_supervised_from_prev_season()
        
        # Train models
        results_df = predictor.train_models()
        
        print("\n" + "="*70)
        print("Model training completed!")
        print("="*70)
        print("\nMODEL PERFORMANCE SUMMARY:")
        print(results_df.to_string(index=False))

        # Generate 2025 player-by-player comparisons and plots
        print("\nGenerating 2025 player-by-player comparisons and accuracy plots...")
        comparison_2025 = predictor.evaluate_on_2025()
        if not comparison_2025.empty:
            print("Saved 'player_predictions_2025.csv' and accuracy plots 'accuracy_2025_*.png'.")
        
        # Optional: interactive predictions for specific players for 2025 season
        try:
            players_to_predict = [s.strip() for s in args.players.split(',') if s.strip()] if args.players else get_user_input()
            if players_to_predict:
                print("\nMaking predictions for specified players (season 2025)...")
                results = []
                test_df = predictor.data[predictor.data['season'] == args.season].copy()
                for name in players_to_predict:
                    subset = test_df[test_df['player'].str.contains(name, case=False, na=False)] if 'player' in test_df.columns else pd.DataFrame()
                    if subset.empty:
                        print(f"\nPlayer '{name}' not found for season {args.season}.")
                        continue
                    row = subset.iloc[[0]]
                    out = {'Player': row['player'].values[0] if 'player' in row.columns else name, 'Season': args.season}
                    print(f"\nPlayer: {out['Player']} (Season {out['Season']})")
                    print(f"{'Stat':<12} {'Actual':<10} {'Predicted':<10} {'Error':<10}")
                    print("-" * 60)
                    for stat, stat_name in predictor.target_stats.items():
                        X = row[predictor.features]
                        X_scaled = predictor.scalers[stat].transform(X)
                        pred = predictor.models[stat].predict(X_scaled)[0]
                        actual = row[stat].values[0] if stat in row.columns else np.nan
                        err = pred - actual if pd.notna(actual) else np.nan
                        print(f"{stat_name:<12} {actual:<10.2f} {pred:<10.2f} {err:<+10.2f}")
                        out[f'Actual_{stat}'] = actual
                        out[f'Predicted_{stat}'] = pred
                        out[f'Error_{stat}'] = err
                    results.append(out)
                if results:
                    pd.DataFrame(results).to_csv('multi_stat_predictions.csv', index=False)
                    print("\nSaved 'multi_stat_predictions.csv' with selected players' predictions.")
        except Exception as e:
            print(f"\nAn error occurred while making player-specific predictions: {str(e)}")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
