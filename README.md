# NBA Player Prop Prediction and Hot Hand Validation

A comprehensive machine learning system for predicting NBA player performance statistics, focusing on points per 36 minutes and multi-stat predictions for the next season.

## 📊 Project Overview

This project implements advanced machine learning models to predict NBA player performance metrics. The system uses historical player data to forecast future season statistics, incorporating features like age, position, games played, and various performance metrics normalized per 36 minutes.

### Key Features
- **Single-Stat Prediction**: Predict points per 36 minutes for the next season
- **Multi-Stat Prediction**: Predict multiple statistics (points, rebounds, assists, steals, blocks)
- **Advanced Feature Engineering**: Lagged features, year-over-year changes, and rolling statistics
- **Multiple ML Models**: Random Forest, XGBoost, and Linear Regression
- **Model Evaluation**: Comprehensive metrics and feature importance analysis
- **Visualization**: Performance plots and feature importance charts

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Required packages listed in `requirements.txt`

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Electrozap/NBA-Player-Prop-Prediction.git
cd NBA-Player-Prop-Prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Data Preparation

The system requires NBA player data in CSV format with the following key columns:
- `player_id`: Unique player identifier
- `player`: Player name
- `season`: Season year
- `age`: Player age
- `pos`: Position
- `g`: Games played
- `mp`: Minutes played
- Performance metrics (pts_per_36_min, trb_per_36_min, ast_per_36_min, etc.)

## 📁 Project Structure

```
├── nba_player_prediction.py     # Main single-stat prediction system
├── nba_multi_stat_predictor.py  # Multi-statistic prediction system
├── predict_players.py          # Player prediction interface
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # Project documentation
```

## 🔧 Usage

### Single-Stat Prediction

Run the main prediction pipeline:

```python
from nba_player_prediction import NBAPlayerPredictor

# Initialize predictor with your data path
predictor = NBAPlayerPredictor('path/to/your/nba_data.csv')

# Run the complete pipeline
predictor.load_and_preprocess()
predictor.create_features()
X_train, X_test, y_train, y_test = predictor.prepare_train_test_split()
predictor.train_models(X_train, y_train)
results = predictor.evaluate_models(X_test, y_test)
```

### Multi-Stat Prediction

For predicting multiple statistics simultaneously:

```python
from nba_multi_stat_predictor import NBAMultiStatPredictor

# Initialize multi-stat predictor
multi_predictor = NBAMultiStatPredictor('path/to/your/nba_data.csv')

# Run predictions for all target stats
predictions = multi_predictor.run_full_pipeline()
```

### Predict Specific Players

Use the prediction interface for individual players:

```python
from predict_players import predict_specific_players

# List of players to predict
players = ["LeBron James", "Stephen Curry", "Giannis Antetokounmpo"]

# Get predictions (requires trained predictor)
results = predict_specific_players(trained_predictor, players)
```

## 🏗️ Architecture

### Feature Engineering
The system creates comprehensive features from historical data:
- **Lagged Features**: Previous season statistics
- **Year-over-Year Changes**: Percentage changes in performance
- **Rolling Statistics**: Moving averages and standard deviations
- **Position Encoding**: One-hot encoded player positions
- **Age and Experience**: Player age and games played

### Machine Learning Models
- **Random Forest**: Ensemble method with feature importance
- **XGBoost**: Gradient boosting with optimized performance
- **Linear Regression**: Baseline model for comparison

### Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Feature importance analysis

## 📈 Model Performance

The system evaluates models using time-series cross-validation to ensure realistic performance estimates. Feature importance plots help identify which statistics are most predictive of future performance.

## 🎯 Key Insights

Based on the model's feature importance analysis, the most predictive features for next season's performance typically include:
- Previous season's points per 36 minutes
- Age and experience metrics
- Year-over-year improvement trends
- Position-specific performance patterns

## 🔍 Data Requirements

### Input Format
The system expects NBA player season-level data with per-36-minute statistics. The data should include:
- Player identifiers and demographics
- Season-by-season performance metrics
- Minutes played and games played
- All traditional basketball statistics

### Data Quality
- Remove players with insufficient playing time (< 500 minutes)
- Handle missing values appropriately
- Ensure consistent season formatting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Data sourced from NBA official statistics
- Built with scikit-learn, XGBoost, and pandas
- Inspired by sports analytics research in basketball performance prediction

## 📞 Contact

For questions or collaboration opportunities:
- GitHub: [@Electrozap](https://github.com/Electrozap)
- Project Link: [https://github.com/Electrozap/NBA-Player-Prop-Prediction](https://github.com/Electrozap/NBA-Player-Prop-Prediction)

---

**Note**: This project is for educational and research purposes. Always verify predictions with domain expertise and consider additional factors not captured in historical data.
