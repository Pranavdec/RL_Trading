from time import time

import numpy as np
import pandas as pd
from datetime import datetime

import gymnasium as gym

# Class for Trading Environment
class TradingEnv(gym.Env):

    def __init__(self, df, df_option_data, window_size, frame_bound, margin, lot_size, options_path = None):
        assert df.ndim == 2
        
        self.frame_bound = frame_bound
        self.options_path = options_path
        self.df = df
        self.df_option_data = df_option_data
        self.df_option_data_grouped = df_option_data.groupby('FH_STRIKE_PRICE')
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()
        self.total_margin = margin
        self.lot_size = lot_size

        # spaces 
        self.action_space = gym.spaces.Tuple((
            gym.spaces.Discrete(4),  # Discrete actions: 4 choices (0, 1, 2, 3) 0 long, 1 short, 2 hold, 3 close
            gym.spaces.Box(low=np.array([1]), high=np.array([5]), dtype=np.float32)  # Continuous 'spread' value
        ))
        self.observation_space = gym.spaces.Dict({
            'stock_data': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size, self.signal_features.shape[1]), dtype=np.float64),
            'strike_prices': gym.spaces.Box(low=0, high=np.inf, shape=(self.window_size, 9), dtype=np.float64),  # ATM, ITM, OTM strike prices history with time to expiry
            'portfolio_info': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(35,8), dtype=np.float64)
        })

        self.trade_fee_buy_percent = 0.01
        self.trade_fee_sell_percent = 0.005
        self.spread = 2 # no.of strikes away from ATM below and above

        # episode
        self._start_tick = self.window_size + frame_bound[0]
        self._end_tick = frame_bound[1]
        self._truncated = None
        self._current_tick = None
        self._current_margin = None
        self._last_trade_tick = None
        self._total_reward = None
        self._total_profit = None
        self._profit_history = None
        self._history = None
        self._portfolio = None
        self._option_data = None
        self._current_margin = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.action_space.seed(int((self.np_random.uniform(0, seed if seed is not None else 1))))
        self._truncated = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._total_reward = 0.
        self._total_profit = 0.
        self._profit_history = [self._total_profit]
        self._history = {}
        self._portfolio = []
        self._current_margin = self.total_margin
        self._option_data = self._get_option_data(self.df.loc[self._current_tick,'Date&Time'])
        self.action_space = gym.spaces.Tuple((
            gym.spaces.Discrete(4),  # Discrete actions: 4 choices (0, 1, 2, 3)
            gym.spaces.Box(low=np.array([1]), high=np.array([5]), dtype=np.float32)  # Continuous 'spread' value
        ))

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # Fetch data relevant to the current tick
        date = self.df.loc[self._current_tick, 'Date&Time']
        date_str  = datetime.strptime(date, '%Y-%m-%d').strftime('%d-%b-%Y')
        strikes = self._process_strike_prices()
        closing_price = self.df.loc[self._current_tick, 'Close']
        atm_index = self._find_nearest_strike_index(closing_price, strikes)
        self.spread = min(int(action[1]), len(strikes) - 1 - atm_index, atm_index)
        itm_index = max(atm_index - self.spread, 0)
        otm_index = min(atm_index + self.spread, len(strikes) - 1)
        atm_strike = strikes[atm_index]
        itm_strike = strikes[itm_index]
        otm_strike = strikes[otm_index]
        atm_price, itm_price, otm_price = self._get_option_prices(atm_strike, itm_strike, otm_strike)
        expiry_date = self.df_option_data[self.df_option_data['FH_TIMESTAMP'] == date_str]['FH_EXPIRY_DT'].iloc[0]

        # Update trading history
        if date not in self._history:
            self._history[date] = {}

        sucess = True
        
        # Handle different actions
        if action[0] == 0 or action == 1:
            sucess = self._add_to_portfolio(action, atm_strike, itm_strike, otm_strike, atm_price, itm_price, otm_price, expiry_date, date_str)

        # Update internal state
        self._update_profit(action)
        step_reward = self._calculate_reward(sucess)
        self._total_reward += step_reward

        # Check and correct invalid reward states
        if np.isinf(self._total_reward) or np.isnan(self._total_reward):
            self._total_reward = 0
            self._total_profit = 1
        
        if self._current_tick == self._end_tick - 1:
            print("End of episode")
            self._truncated = True

        # Prepare for next step
        self._last_trade_tick = self._current_tick
        self._current_tick = self._current_tick + 1
        observation = self._get_observation()
        info = self._get_info()

        # print(f"Portfolio: {self.portfolio}")
        return observation, step_reward, False, self._truncated, info

    def _process_strike_prices(self):
        x = self.df.loc[self._current_tick, 'Strike Prices']
        y = len(x) - 1
        x = x[1:y]
        list_1 = x.split(',')
        list_1 = [float(i) for i in list_1]
        return list_1
    
    def _find_nearest_strike_index(self, target_price, strike_prices):
        strike_prices = list(strike_prices)
        nearest_index = np.abs(strike_prices - target_price).argmin()
        return nearest_index   
        
    def _get_option_prices(self, atm_strike, itm_strike, otm_strike):
        atm_price = self._get_option_price(atm_strike)
        itm_price = self._get_option_price(itm_strike)
        otm_price = self._get_option_price(otm_strike)
        return atm_price, itm_price, otm_price
        
    def _get_option_price(self, strike_price):
        date = self.df.loc[self._current_tick,'Date&Time']
        date = datetime.strptime(date, '%Y-%m-%d').strftime('%d-%b-%Y')
        price = self.df_option_data[(self.df_option_data['FH_STRIKE_PRICE'] == strike_price) & (self.df_option_data['FH_TIMESTAMP'] == date)]['FH_CLOSING_PRICE'].iloc[0]
        return price  

    def _add_to_portfolio(self, action, atm_strike, itm_strike, otm_strike, atm_price, itm_price, otm_price, expiry_date, date):
        # Define trade type based on action
        trade_type = 'long' if action == 0 else 'short'

        # Initialize total cost and margin
        total_cost = 0
        margin = 0

        # Adjust cost and margin based on trade type
        if trade_type == 'short':
            margin -= self._calculate_margin_sell(itm_price) * self.lot_size
            margin -= self._calculate_margin_sell(otm_price) * self.lot_size
            margin -= 2 * atm_price * self.lot_size

            total_cost -= itm_price * self.lot_size * (self.trade_fee_buy_percent)
            total_cost -= otm_price * self.lot_size * (self.trade_fee_buy_percent)
            total_cost -= 2 * atm_price * self.lot_size * (self.trade_fee_buy_percent)

        elif trade_type == 'long':
            margin -= self._calculate_margin_sell(atm_price) * self.lot_size * 2
            margin -= itm_price * self.lot_size
            margin -= otm_price * self.lot_size

            total_cost -= itm_price * self.lot_size * (self.trade_fee_buy_percent)
            total_cost -= otm_price * self.lot_size * (self.trade_fee_buy_percent)
            total_cost -= atm_price * self.lot_size * (self.trade_fee_buy_percent) * 2

        # Update total cost with calculated margin
        total_cost += margin

        # Check if the trade is possible with current margin
        if abs(total_cost) > self._current_margin:
            return False

        # Record the trade in portfolio and history
        self.portfolio.append((trade_type, atm_strike, itm_strike, otm_strike, atm_price, itm_price, otm_price, expiry_date, abs(margin)))
        self.history[date][trade_type] = (atm_strike, itm_strike, otm_strike, atm_price, itm_price, otm_price, expiry_date, abs(margin))

        # Update current margin
        self._current_margin += total_cost
        return True
       
    def _calculate_margin_sell(self, price):
        current_price = self.df[self.df['Date&Time'] == self.df.loc[self._current_tick, 'Date&Time']]['Close'].iloc[0]
        date = self.df.loc[self._current_tick, 'Date&Time']
        date = datetime.strptime(date, '%Y-%m-%d').strftime('%d-%b-%Y')
        otm_price = self.df_option_data[self.df_option_data['FH_TIMESTAMP'] == date]['FH_CLOSING_PRICE'].iloc[-1]
        margin = price + current_price * 0.2 - otm_price
        margin1 = price + current_price * 0.1
        return max(margin, margin1)
        
    def _get_info(self):
        return dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            drawdown = self._calculate_maximum_drawdown(self._profit_history)
        )

    def _get_observation(self):
        output = {}
        
        # Calculate the current tick within the local frame
        x = self._current_tick - self.frame_bound[0]
        
        # Get the current tick's signal features (e.g., stock data)
        current_features = self.signal_features[(x - self.window_size + 1) : x + 1]
        output['stock_data'] = current_features

        # Process the options data for the current tick
        output['strike_prices'] = self._process_option_data_for_observation()

        # Assume some method exists to gather portfolio info
        output['portfolio_info'] = self._process_portfolio_info()

        return output

    def _process_option_data_for_observation(self):
        option_data = self.df_option_data_grouped

        strikes = self._process_strike_prices()
        closing_price = self.df.loc[self._current_tick, 'Close']
        atm_index = self._find_nearest_strike_index(closing_price, strikes)
        itm_index = max(atm_index - self.spread, 0)
        otm_index = min(atm_index + self.spread, len(strikes) - 1)

        # Extract strike prices for ATM, ITM, OTM
        atm_strike = strikes[atm_index]
        itm_strike = strikes[itm_index]
        otm_strike = strikes[otm_index]

        # Get option data for these strikes
        atm_data = option_data.get_group(atm_strike)
        itm_data = option_data.get_group(itm_strike)
        otm_data = option_data.get_group(otm_strike)

        # Filter the data by the relevant dates
        window_start = max(self._current_tick - self.window_size, 0)
        date_window = self.df.loc[window_start:self._current_tick, 'Date&Time'].unique()
        formatted_date_window = [datetime.strptime(date, '%Y-%m-%d').strftime('%d-%b-%Y') for date in date_window]

        # Filter each type's data to include only those dates
        atm_data = atm_data[atm_data['FH_TIMESTAMP'].isin(formatted_date_window)]
        itm_data = itm_data[itm_data['FH_TIMESTAMP'].isin(formatted_date_window)]
        otm_data = otm_data[otm_data['FH_TIMESTAMP'].isin(formatted_date_window)]

        # Initialize arrays to store the data with shape (window_size, 3)
        atm_info = np.zeros((self.window_size, 3))
        itm_info = np.zeros((self.window_size, 3))
        otm_info = np.zeros((self.window_size, 3))

        # Fill the arrays with available data
        for i, date in enumerate(formatted_date_window[-self.window_size:]):
            if date in atm_data['FH_TIMESTAMP'].values:
                atm_info[i] = atm_data.loc[atm_data['FH_TIMESTAMP'] == date, ['FH_CLOSING_PRICE', 'FH_TOT_TRADED_QTY', 'Days_to_Expiry']].values
            if date in itm_data['FH_TIMESTAMP'].values:
                itm_info[i] = itm_data.loc[itm_data['FH_TIMESTAMP'] == date, ['FH_CLOSING_PRICE', 'FH_TOT_TRADED_QTY', 'Days_to_Expiry']].values
            if date in otm_data['FH_TIMESTAMP'].values:
                otm_info[i] = otm_data.loc[otm_data['FH_TIMESTAMP'] == date, ['FH_CLOSING_PRICE', 'FH_TOT_TRADED_QTY', 'Days_to_Expiry']].values

        # Concatenate all info into a single output array
        output = np.hstack([atm_info, itm_info, otm_info])

        return output

    def _process_portfolio_info(self):
        # Initialize arrays to store the portfolio info
        portfolio_info = np.zeros((35, 8))
        
        current_date_str = datetime.strptime(self.df.loc[self._current_tick, 'Date&Time'], '%Y-%m-%d')

        # Fill the arrays with available data
        for i, position in enumerate(self._portfolio):
            position = list(position)
            if position[0] == 'long':
                position[0] = 1
            else:
                position[0] = -1
                
            date = datetime.strptime(position[7], '%d-%b-%Y')
            days_to_expiry = (date - current_date_str).days + 1
            position[7] = days_to_expiry
            
            portfolio_info[i] = position

        return portfolio_info

    def _get_option_data(self,date):
        # get the option data for the given date
        date = datetime.strptime(date, '%Y-%m-%d').strftime('%d-%b-%Y')
        path = self.options_path + "/" + date + ".csv"
        option_data = pd.read_csv(path)
        return option_data
    
    def find_nearest_strikeprice_onexpiry(self, target_price, date):
        # find the nearest strike price to the target price on the expiry date
        date = date.strftime('%Y-%m-%d')
        option_data = self._get_option_data(date)
        nearest_index = (option_data['FH_STRIKE_PRICE'] - target_price).abs().argmin()
        price = option_data.loc[nearest_index]['FH_CLOSING_PRICE']
        return price
    
    def _calculate_maximum_drawdown(self, cumulative_returns):
        # Calculate the high water marks
        highwater_marks = np.maximum.accumulate(cumulative_returns)
        
        # Initialize drawdowns with zeros
        drawdowns = np.zeros_like(cumulative_returns)
        
        # Find indices where high water marks are greater than zero
        valid_indices = [i for i, hw in enumerate(highwater_marks) if hw > 0]
        
        # Calculate drawdowns only for valid indices
        for i in valid_indices:
            drawdowns[i] = (cumulative_returns[i] - highwater_marks[i]) / highwater_marks[i]
        
        # Find the maximum drawdown
        max_drawdown = drawdowns.min()
        return max_drawdown
    
    def _process_data(self):
        # create the signal features
        prices = self.df.loc[:, 'Close'].to_numpy()

        features = {
            'volume': self.df.loc[:, 'Volume'].to_numpy(),
            'week_high_52': self.df.loc[:, '52_Week_High'].to_numpy(),
            'week_low_52': self.df.loc[:, '52_Week_Low'].to_numpy(),
            'rsi': self.df.loc[:, 'RSI'].to_numpy(),
            'macd': self.df.loc[:, 'MACD'].to_numpy(),
            'macd_signal': self.df.loc[:, 'Signal'].to_numpy(),
            'upper_band': self.df.loc[:, 'BB_Upper'].to_numpy(),
            'lower_band': self.df.loc[:, 'BB_Lower'].to_numpy(),
            'volatility': self.df.loc[:, 'Volatility'].to_numpy(),
            'stochastic_k': self.df.loc[:, '%K'].to_numpy(),
            'stochastic_d': self.df.loc[:, '%D'].to_numpy(),
            'obv': self.df.loc[:, 'OBV'].to_numpy(),
            'apo': self.df.loc[:, 'APO'].to_numpy(),
            'ppo': self.df.loc[:, 'PPO'].to_numpy(),
            'ppo_signal': self.df.loc[:, 'PPO_Signal'].to_numpy(),
            'smas': self.df.loc[:, 'SMA'].to_numpy(),
            'emas': self.df.loc[:, 'EMA'].to_numpy(),
            'iv': self.df.loc[:, 'Implied_Volatility'].to_numpy(),
            'Delta': self.df.loc[:, 'Delta'].to_numpy(),
            'Gamma': self.df.loc[:, 'Gamma'].to_numpy(),
            'Theta': self.df.loc[:, 'Theta'].to_numpy(),
            'Vega': self.df.loc[:, 'Vega'].to_numpy()
        }

        # Ensure all feature arrays match the length of prices
        for key in features:
            features[key] = features[key][self.frame_bound[0]:self.frame_bound[1]]

        prices = prices[self.frame_bound[0]:self.frame_bound[1]]
        
        # Compute the difference of prices to use as an additional feature
        diff = np.insert(np.diff(prices), 0, 0)

        # Stack all features including prices and diff
        signal_features = np.column_stack([prices, diff] + [features[key] for key in features])

        return prices.astype(np.float64), signal_features.astype(np.float64)

    def _calculate_reward(self,sucess):
        profits = self._profit_history
        if not profits:
            return 0

        # Initialize variables
        current_streak = 0
        last_sign = None
        total_reward = 0

        # Calculate the reward based on profit streaks
        for profit in profits:
            if profit != 0 :
                current_sign = (profit > 0)
                
                if current_sign == last_sign:
                    current_streak += 1  # Continue the streak
                else:
                    # If streak ends, calculate reward for that streak
                    if current_streak > 0:
                        streak_multiplier = 1.1 ** current_streak
                        total_reward += streak_multiplier * (sum(profits[-current_streak:]) if last_sign else -sum(profits[-current_streak:]))
                    current_streak = 1  # Reset streak count
                
                last_sign = current_sign

        # Add reward for the last streak if it's positive
        if current_streak > 0 and last_sign:
            streak_multiplier = 1.1 ** current_streak
            total_reward += streak_multiplier * sum(profits[-current_streak:])
            
        max_drawdown = self._calculate_maximum_drawdown(profits)
        
        total_reward = 0.7 * profits[-1] + 0.3 * total_reward - 0.5 * max_drawdown
        
        if sucess == False:
            total_reward -= 10

        # Apply a simple cap to the total_reward
        total_reward = min(max(total_reward, -100), 100)  # Clamp the reward between -100 and 100
        return total_reward

    def _calculate_profit(self, old_prices, new_prices, option_type):
        sum = 0
        if option_type == 'long':
            sum += (new_prices[0] - old_prices[0]) * 2
            sum += (old_prices[1] - new_prices[1])
            sum += (old_prices[2] - new_prices[2])
        else:
            sum += (old_prices[0] - new_prices[0]) * 2
            sum += (new_prices[1] - old_prices[1])
            sum += (new_prices[2] - old_prices[2])
        return sum

    def _update_profit(self,action):
        final_profit = 0
        expired_options = []
        current_date_str = self.df.loc[self._current_tick, 'Date&Time']
        
        if action == 3:
            for i in range(len(self._portfolio)):
                position = self._portfolio[i]
                expired_options.append(i)
                option_type = position[0]
                strikes = position[1:4]
                prices = position[4:7]
                
                new_prices = [self.find_nearest_strikeprice_onexpiry(strike, current_date_str) for strike in strikes]
                profit = self._calculate_profit(prices, new_prices, option_type)
                final_profit += profit

        # Check and update profit for expired options
        for i in range(len(self._portfolio) - 1, -1, -1):
            if i not in expired_options:
                expiry_date_str = self._portfolio[i][7]
                current_date = datetime.strptime(current_date_str, '%Y-%m-%d')
                expiry_date = datetime.strptime(expiry_date_str, '%d-%b-%Y')

                if current_date > expiry_date:
                    position = self._portfolio[i]
                    expired_options.append(i)     
                    option_type = position[0]
                    strikes = position[1:4]
                    prices = position[4:7]
                    
                    new_prices = [self.find_nearest_strikeprice_onexpiry(strike, expiry_date) for strike in strikes]
                    profit = self._calculate_profit(prices, new_prices, option_type)
                    final_profit += profit
        
        # Remove expired options from the portfolio
        expired_options.sort(reverse=True)
        
        for i in expired_options:
            x = self._portfolio.pop(i)
            margin = x[-1]
            self._current_margin += margin

        # Update total profit and profit history
        self._total_profit += final_profit
        self._current_margin += final_profit
        self._profit_history.append(final_profit)
        