"""
Predict next season's points per 36 minutes for specific NBA players
"""

import pandas as pd
import numpy as np
from nba_player_prediction import NBAPlayerPredictor

def predict_specific_players(predictor, player_names):
    """
    Predict next season's performance for specific players.
    
    Args:
        predictor: Trained NBAPlayerPredictor instance
        player_names: List of player names to predict
    """
    results = []
    
    for player_name in player_names:
        # Find the player's most recent season data
        player_data = predictor.data[predictor.data['player'].str.contains(player_name, case=False, na=False)]
        
        if player_data.empty:
            print(f"\n❌ Player '{player_name}' not found in the dataset.")
            continue
        
        # Get the most recent season for this player
        latest_season = player_data.sort_values('season', ascending=False).iloc[0:1]
        current_season = int(latest_season['season'].values[0])
        predicted_season = current_season + 1
        
        print(f"\n{'='*60}")
        print(f"🏀 Player: {latest_season['player'].values[0]}")
        print(f"{'='*60}")
        print(f"Most Recent Season: {current_season}")
        print(f"Predicting For Season: {predicted_season}")
        print(f"Age: {latest_season['age'].values[0]}")
        print(f"Team: {latest_season['team'].values[0]}")
        print(f"Position: {latest_season['pos'].values[0]}")
        print(f"Games Played: {latest_season['g'].values[0]}")
        print(f"Current Points per 36 min: {latest_season['pts_per_36_min'].values[0]:.2f}")
        
        try:
            # Make prediction using Random Forest model only
            predictions = predictor.predict_player_season(latest_season)
            
            if 'Random Forest' in predictions:
                rf_pred = predictions['Random Forest']['prediction']
                current_pts = predictions['Random Forest']['current_pts_per_36']
                change = rf_pred - current_pts if current_pts else None
                
                print(f"\n📊 RANDOM FOREST PREDICTION:")
                print(f"   Predicted Points per 36 min ({predicted_season} Season): {rf_pred:.2f}")
                if change is not None:
                    print(f"   Expected Change: {change:+.2f} points")
                    if change > 0:
                        print(f"   📈 Trend: IMPROVING ({(change/current_pts)*100:+.1f}%)")
                    elif change < 0:
                        print(f"   📉 Trend: DECLINING ({(change/current_pts)*100:+.1f}%)")
                    else:
                        print(f"   ➡️  Trend: STABLE")
                
                results.append({
                    'Player': latest_season['player'].values[0],
                    'Current_Season': current_season,
                    'Predicted_Season': predicted_season,
                    'Age': latest_season['age'].values[0],
                    'Current_PTS_36': current_pts,
                    'Predicted_PTS_36': rf_pred,
                    'Change': change
                })
        
        except Exception as e:
            print(f"\n❌ Error making prediction: {str(e)}")
    
    return pd.DataFrame(results)

def get_user_input():
    """
    Get player names and prediction preferences from user input.
    """
    print("\n" + "="*60)
    print("🏀 NBA PLAYER PERFORMANCE PREDICTOR".center(60))
    print("="*60)
    
    # Get player names
    print("\nEnter player names to predict (separated by commas):")
    print("Example: Nikola Jokic, Bam Adebayo, LeBron James")
    player_input = input("Players: ").strip()
    
    if not player_input:
        print("⚠️  No players entered. Using default players.")
        players = ['Nikola Jokic', 'Bam Adebayo']
    else:
        players = [name.strip() for name in player_input.split(',')]
    
    print(f"\n✅ Will predict for: {', '.join(players)}")
    
    return players

def main():
    # Initialize and train the predictor
    print("Initializing NBA Player Predictor...")
    predictor = NBAPlayerPredictor(r'C:\Users\VICTUS\OneDrive\Desktop\SEM V\ML\NBA_36.csv')
    
    try:
        # Load and preprocess data
        print("Loading and preprocessing data...")
        predictor.load_and_preprocess()
        
        # Create features
        print("Creating features...")
        predictor.create_features()
        
        # Prepare train/test split
        print("Preparing train/test split...")
        X_train, X_test, y_train, y_test = predictor.prepare_train_test_split()
        
        # Train models
        print("Training models...")
        predictor.train_models(X_train, y_train)
        
        print("\n" + "="*60)
        print("✅ Model training completed!")
        print("="*60)
        
        # Get user input for players to predict
        players_to_predict = get_user_input()
        
        print("\n\n" + "🎯 MAKING PREDICTIONS FOR SPECIFIC PLAYERS".center(60))
        results_df = predict_specific_players(predictor, players_to_predict)
        
        if not results_df.empty:
            print("\n\n" + "="*60)
            print("📋 SUMMARY TABLE")
            print("="*60)
            print(results_df.to_string(index=False))
            
            # Save results to CSV
            results_df.to_csv('player_predictions.csv', index=False)
            print("\n💾 Results saved to 'player_predictions.csv'")
        
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
