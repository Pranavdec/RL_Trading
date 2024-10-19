# Punjab National Bank Options Trading Strategy Implementation

## Overview
This project implements a butterfly trading strategy for Punjab National Bank (PNB) stock options. The approach includes data collection, preprocessing, model training, and trading simulation based on historical data.

## Setup Instructions

### Prerequisites
- Python 3.8+

### Installation
1. Clone the repository:
   ```bash
   [git clone https://github.com/<your-username>/pnb-options-strategy.git](https://github.com/Pranavdec/RL_Trading.git)
   ```
2. Navigate to the project directory:
   ```bash
   cd RL_Trading
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Collection and Preprocessing

The data for this project is already provided in the `data/` directory. If you wish to collect fresh data:
1. Navigate to the `notebooks/` directory.
2. Run `CollectionStockData.ipynb` using the Angel One Smart API to collect stock data.
3. Run `CollectionOptionData.ipynb` to collect options data.
4. Use `Preprocessing.ipynb` to preprocess the collected data.

## Model Training and Simulation
1. Navigate to the `src/` directory.
2. Run the trading environment scripts and model training scripts as documented in each file.

## Project Structure
- **data/**: Contains raw and preprocessed data files.
- **notebooks/**: Jupyter notebooks for data collection and preprocessing.
- **src/**: Python scripts for the trading environments and models.
- **models/**: Trained model files and configuration.
- **requirements.txt**: Required libraries.

## Limitations
- The trading models have not been tested with varying margin requirements.
- Profit calculations are based on single lot trades. This limitation must be considered when interpreting financial outcomes.

### Key Points to Note:
1. **Data Collection**: It's specified that the data is already provided but can be refreshed using the given notebooks and an API.
2. **Project Structure**: Clearly defined to help new users navigate through the repository.
3. **Limitations**: Noted clearly to set proper expectations for users.
4. **Installation and Setup**: Detailed steps ensure users can get the project running without prior setup knowledge.

## Detailed Steps for Running Jupyter Notebooks

### Order of Execution
To run the project successfully, follow these steps in the given order, depending on the scenario you wish to simulate:

1. **Data Collection**:
   - Run `CollectionStockData.ipynb` to gather historical stock data. This notebook uses the Angel One Smart API but you can modify it to use other data sources.
   - Run `CollectionOptionData.ipynb` to collect options data corresponding to the collected stock data.

2. **Data Preprocessing**:
   - Run `Preprocessing.ipynb` to process both stock and options data by calculating various financial indicators and metrics which are crucial for the trading strategies.

### Trading Environment Details

Three trading environments (`trading_env_final.py`, `trading_env1.py`, and `trading_env.py`) are designed to simulate different trading scenarios:

1. **trading_env_final.py**:
   - This environment focuses on basic butterfly trading strategies with actions like Long Butterfly, Short Butterfly, Hold, and Close.
   - It uses a discrete action space and factors in trading dynamics like transaction fees, margin requirements, and option expiration handling.

2. **trading_env1.py**:
   - Similar to the final environment but allows for more dynamic adjustments to the butterfly spread strategy, accommodating market conditions.
   - It expands the action space to include Increase Spread and Decrease Spread actions, providing more control over the trading positions.

3. **trading_env.py**:
   - Incorporates both discrete and continuous action spaces, where discrete actions are basic trading decisions and the continuous component allows for precise spread value selection.
   - The environment processes each trading tick with detailed market and portfolio information, allowing for a highly customizable trading simulation.

### Testing and Model Selection

The testing phase involved the following steps:
- Models were trained using the A2C, DQN, and PPO algorithms across multiple scenarios to determine the most effective strategy based on profit and drawdown metrics.
- The A2C algorithm with a spread of 5 was selected for further hyperparameter tuning due to its superior performance in initial tests.

### Hyperparameter Tuning and Final Model Performance
- Optuna was used for hyperparameter tuning, and the best-performing parameters were applied to the final model.
- The model's performance, profitability, and risk metrics were carefully documented and analyzed, considering the simulation constraints and trading fees.

## Contributing
We welcome contributions to enhance the project's robustness and feature set. For major changes, please open an issue first to discuss what you would like to change.
