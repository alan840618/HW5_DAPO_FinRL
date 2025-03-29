from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv

matplotlib.use("Agg")


class DAPOInferenceEnv(gym.Env):
    """A custom environment for DAPO inference with LLM risk and sentiment features
    
    State Space Structure:
    1. Cash balance [1 value]
    2. Current stock prices [stock_dimension values]
    3. Current holdings [stock_dimension values]
    4. Tradable flags [stock_dimension values]
    5. Technical indicators [len(tech_indicator_list)*stock_dimension values]
    6. LLM sentiment scores [stock_dimension values]
    7. LLM risk scores [stock_dimension values]
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int,
        initial_amount: int,
        num_stock_shares: list[int],
        buy_cost_pct: list[float],
        sell_cost_pct: list[float],
        reward_scaling: float,
        state_space: int,
        action_space: int,
        tech_indicator_list: list[str],
        turbulence_threshold=None,
        risk_indicator_col="vix",
        llm_sentiment_col="llm_sentiment",
        llm_risk_col="llm_risk",
        make_plots: bool = False,
        print_verbosity=10,
        day=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
    ):
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.num_stock_shares = num_stock_shares
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.llm_sentiment_col = llm_sentiment_col
        self.llm_risk_col = llm_risk_col
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        
        # Initialize state
        self.state = self._initiate_state()

        # Initialize tracking variables
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        
        # Memory for tracking performance
        self.asset_memory = [
            self.initial_amount
            + np.sum(
                np.array(self.num_stock_shares)
                * np.array(self.state[1 : 1 + self.stock_dim])
            )
        ]
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = []
        self.date_memory = [self._get_date()]
        self.portfolio_memory = []
        
        # Initialize random seed
        self.seed()

    def _sell_stock(self, index, action):
        """Execute a sell stock action"""
        def _do_sell_normal():
            if self.state[index + 2 * self.stock_dim + 1] != True:  # Check if stock is tradable
                if self.state[index + self.stock_dim + 1] > 0:  # If we have shares to sell
                    sell_num_shares = min(abs(action), self.state[index + self.stock_dim + 1])
                    sell_amount = (
                        self.state[index + 1]
                        * sell_num_shares
                        * (1 - self.sell_cost_pct[index])
                    )
                    
                    # Update balance and holdings
                    self.state[0] += sell_amount
                    self.state[index + self.stock_dim + 1] -= sell_num_shares
                    self.cost += (
                        self.state[index + 1]
                        * sell_num_shares
                        * self.sell_cost_pct[index]
                    )
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares

        # Handle turbulence if threshold is set
        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                if self.state[index + 1] > 0:  # If price is valid
                    if self.state[index + self.stock_dim + 1] > 0:  # If we have shares
                        # Sell all shares during high turbulence
                        sell_num_shares = self.state[index + self.stock_dim + 1]
                        sell_amount = (
                            self.state[index + 1]
                            * sell_num_shares
                            * (1 - self.sell_cost_pct[index])
                        )
                        self.state[0] += sell_amount
                        self.state[index + self.stock_dim + 1] = 0
                        self.cost += (
                            self.state[index + 1]
                            * sell_num_shares
                            * self.sell_cost_pct[index]
                        )
                        self.trades += 1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):
        """Execute a buy stock action"""
        def _do_buy():
            # Updated index for tradable flag  
            if self.state[index + 2 * self.stock_dim + 1] != True:  # Check if stock is tradable
                # Calculate available amount considering trading costs
                available_amount = self.state[0] // (
                    self.state[index + 1] * (1 + self.buy_cost_pct[index])
                )
                
                # Execute buy action
                buy_num_shares = min(available_amount, action)
                buy_amount = (
                    self.state[index + 1]
                    * buy_num_shares
                    * (1 + self.buy_cost_pct[index])
                )
                self.state[0] -= buy_amount
                self.state[index + self.stock_dim + 1] += buy_num_shares
                self.cost += (
                    self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index]
                )
                self.trades += 1
            else:
                buy_num_shares = 0

            return buy_num_shares

        # Check turbulence before buying
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0

        return buy_num_shares

    def _make_plot(self):
        """Create a plot of portfolio performance"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.date_memory, self.asset_memory, 'r', label='Portfolio Value')
        plt.title(f"DAPO Portfolio Performance - Episode {self.episode}")
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"dapo_results/portfolio_value_episode_{self.episode}.png", dpi=300)
        plt.close()

    def step(self, actions):
        """Execute one step in the environment"""
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        
        # If terminal state, calculate final performance metrics
        if self.terminal:
            if self.make_plots:
                self._make_plot()
                
            # Calculate final portfolio value
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            
            # Create dataframe of portfolio values over time
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(1)
            
            # Calculate Sharpe ratio if possible
            sharpe = 0
            if df_total_value["daily_return"].std() != 0:
                sharpe = (
                    (252**0.5)
                    * df_total_value["daily_return"].mean()
                    / df_total_value["daily_return"].std()
                )
                
            # Calculate total reward
            tot_reward = end_total_asset - self.asset_memory[0]
            
            # Print summary if verbose
            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if df_total_value["daily_return"].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

            # Save results if model name and mode are specified
            if (self.model_name != "") and (self.mode != ""):
                # Save actions
                df_actions = self.save_action_memory()
                df_actions.to_csv(
                    f"dapo_results/actions_{self.mode}_{self.model_name}_{self.iteration}.csv"
                )
                
                # Save portfolio values
                df_total_value.to_csv(
                    f"dapo_results/account_value_{self.mode}_{self.model_name}_{self.iteration}.csv",
                    index=False,
                )
                
                # Save rewards
                df_rewards = pd.DataFrame(self.rewards_memory)
                df_rewards.columns = ["account_rewards"]
                df_rewards["date"] = self.date_memory[:-1]
                df_rewards.to_csv(
                    f"dapo_results/account_rewards_{self.mode}_{self.model_name}_{self.iteration}.csv",
                    index=False,
                )
                
                # Save portfolio allocation
                if self.portfolio_memory:
                    df_portfolio = pd.DataFrame(self.portfolio_memory)
                    df_portfolio.to_csv(
                        f"dapo_results/portfolio_allocation_{self.mode}_{self.model_name}_{self.iteration}.csv",
                        index=False,
                    )

            return self.state, self.reward, self.terminal, False, {}

        else:
            # Process actions for the current step
            
            # Apply LLM sentiment to influence actions
            llm_sentiments = self.data[self.llm_sentiment_col].values
            llm_risks = self.data[self.llm_risk_col].values
            
            # Create masks for action types
            buy_mask = (actions > 0)
            sell_mask = (actions < 0)
            
            # Create masks based on LLM sentiments (1-5 scale)
            strong_sell_mask = (llm_sentiments == 1)
            moderate_sell_mask = (llm_sentiments == 2)
            hold_mask = (llm_sentiments == 3)
            moderate_buy_mask = (llm_sentiments == 4)
            strong_buy_mask = (llm_sentiments == 5)
            
            # Adjust actions based on LLM sentiment
            # Reduce actions that go against sentiment
            actions[(strong_sell_mask & buy_mask) | (strong_buy_mask & sell_mask)] *= 0.9
            actions[(moderate_sell_mask & buy_mask) | (moderate_buy_mask & sell_mask)] *= 0.95
            
            # Amplify actions that align with sentiment
            actions[(strong_sell_mask & sell_mask) | (strong_buy_mask & buy_mask)] *= 1.1
            actions[(moderate_sell_mask & sell_mask) | (moderate_buy_mask & buy_mask)] *= 1.05
            
            # Get LLM risk scores for the current day
            llm_risks = self.data[self.llm_risk_col].values
            
            # Convert actions to numpy array if not already
            actions = np.array(actions).flatten()
            
            # Create risk masks
            high_risk_mask = (llm_risks >= 4)
            low_risk_mask = (llm_risks <= 2)
            
            # Reduce position sizes in high risk assets
            actions = np.where(high_risk_mask, actions * 0.9, actions)
            
            # Scale actions to match hmax
            actions = actions * self.hmax
            actions = actions.astype(int)
            
            # Force selling during high turbulence if threshold is set
            if self.turbulence_threshold is not None:
                if self.turbulence >= self.turbulence_threshold:
                    actions = np.array([-self.hmax] * self.stock_dim)
            
            # Record beginning portfolio value
            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            
            # Sort actions by value
            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]
            
            # Execute sell actions first (to free up cash)
            for index in sell_index:
                actions[index] = self._sell_stock(index, actions[index]) * (-1)
            
            # Execute buy actions
            for index in buy_index:
                actions[index] = self._buy_stock(index, actions[index])
            
            # Record actions
            self.actions_memory.append(actions)
            
            # Record portfolio allocation
            portfolio_allocation = {}
            for i in range(self.stock_dim):
                stock_name = self.data.tic.values[i] if len(self.df.tic.unique()) > 1 else self.data.tic
                shares = self.state[i + self.stock_dim + 1]
                price = self.state[i + 1]
                portfolio_allocation[stock_name] = shares * price
            portfolio_allocation['cash'] = self.state[0]
            self.portfolio_memory.append(portfolio_allocation)
            
            # Move to next day
            self.day += 1
            self.data = self.df.loc[self.day, :]
            
            # Update turbulence if applicable
            if self.turbulence_threshold is not None:
                if len(self.df.tic.unique()) == 1:
                    self.turbulence = self.data[self.risk_indicator_col]
                elif len(self.df.tic.unique()) > 1:
                    self.turbulence = self.data[self.risk_indicator_col].values[0]
            
            # Update state
            self.state = self._update_state()
            
            # Calculate reward and update memories
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            self.reward = end_total_asset - begin_total_asset
            self.rewards_memory.append(self.reward)
            self.reward = self.reward * self.reward_scaling
            self.state_memory.append(self.state)

        return self.state, self.reward, self.terminal, False, {}

    def reset(self, *, seed=None, options=None):
        """Reset the environment"""
        # Reset to day 0
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.state = self._initiate_state()

        # Reset asset memory
        if self.initial:
            self.asset_memory = [
                self.initial_amount
                + np.sum(
                    np.array(self.num_stock_shares)
                    * np.array(self.state[1 : 1 + self.stock_dim])
                )
            ]
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(
                    self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                )
            )
            self.asset_memory = [previous_total_asset]

        # Reset other tracking variables
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self.portfolio_memory = []
        self.episode += 1

        return self.state, {}

    def render(self, mode="human", close=False):
        """Render the environment"""
        return self.state

    def _initiate_state(self):
        """Initialize the state"""
        if self.initial:
            # For Initial State
            if len(self.df.tic.unique()) > 1:
                # For multiple stocks
                state = (
                    [self.initial_amount]  # Cash
                    + self.data.close.values.tolist()  # Current prices
                    + self.num_stock_shares  # Current holdings
                    + [False] * self.stock_dim  # Tradable flags (all tradable by default)
                    + sum((self.data[tech].values.tolist() for tech in self.tech_indicator_list), [])  # Technical indicators
                    + self.data[self.llm_sentiment_col].values.tolist()  # LLM sentiment
                    + self.data[self.llm_risk_col].values.tolist()  # LLM risk
                )
            else:
                # For single stock
                state = (
                    [self.initial_amount]  # Cash
                    + [self.data.close]  # Current price
                    + [self.num_stock_shares[0]]  # Current holding (single value)
                    + [False]  # Tradable flag (tradable by default)
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])  # Technical indicators
                    + [self.data[self.llm_sentiment_col]]  # LLM sentiment
                    + [self.data[self.llm_risk_col]]  # LLM risk
                )
        else:
            # Using Previous State
            if len(self.df.tic.unique()) > 1:
                # For multiple stocks
                state = (
                    [self.previous_state[0]]  # Cash from previous state
                    + self.data.close.values.tolist()  # Current prices
                    + self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]  # Holdings from previous state
                    + self.previous_state[(self.stock_dim * 2 + 1) : (self.stock_dim * 3 + 1)]  # Tradable flags from previous state
                    + sum((self.data[tech].values.tolist() for tech in self.tech_indicator_list), [])  # Technical indicators
                    + self.data[self.llm_sentiment_col].values.tolist()  # LLM sentiment
                    + self.data[self.llm_risk_col].values.tolist()  # LLM risk
                )
            else:
                # For single stock
                state = (
                    [self.previous_state[0]]  # Cash from previous state
                    + [self.data.close]  # Current price
                    + [self.previous_state[self.stock_dim + 1]]  # Holding from previous state (single value)
                    + [self.previous_state[self.stock_dim * 2 + 1]]  # Tradable flag from previous state
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])  # Technical indicators
                    + [self.data[self.llm_sentiment_col]]  # LLM sentiment
                    + [self.data[self.llm_risk_col]]  # LLM risk
                )

        return state

    def _update_state(self):
        """Update the state"""
        if len(self.df.tic.unique()) > 1:
            # For multiple stocks
            state = (
                [self.state[0]]  # Cash
                + self.data.close.values.tolist()  # Current prices
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])  # Current holdings
                + list(self.state[(self.stock_dim * 2 + 1) : (self.stock_dim * 3 + 1)])  # Tradable flags
                + sum((self.data[tech].values.tolist() for tech in self.tech_indicator_list), [])  # Technical indicators
                + self.data[self.llm_sentiment_col].values.tolist()  # LLM sentiment
                + self.data[self.llm_risk_col].values.tolist()  # LLM risk
            )
        else:
            # For single stock
            state = (
                [self.state[0]]  # Cash
                + [self.data.close]  # Current price
                + [self.state[self.stock_dim + 1]]  # Current holding (single value)
                + [self.state[self.stock_dim * 2 + 1]]  # Tradable flag
                + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])  # Technical indicators
                + [self.data[self.llm_sentiment_col]]  # LLM sentiment
                + [self.data[self.llm_risk_col]]  # LLM risk
            )

        return state

    def _get_date(self):
        """Get the current date"""
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

    def save_asset_memory(self):
        """Save the asset memory to a DataFrame"""
        date_list = self.date_memory
        asset_list = self.asset_memory
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value

    def save_action_memory(self):
        """Save the action memory to a DataFrame"""
        try:
            if len(self.df.tic.unique()) > 1:
                # For multiple stocks
                date_list = self.date_memory[:-1]
                
                # Check if date_list is empty
                if len(date_list) == 0:
                    print("Warning: Empty date list in save_action_memory. Returning empty DataFrame.")
                    return pd.DataFrame()
                
                df_date = pd.DataFrame(date_list)
                df_date.columns = ["date"]

                action_list = self.actions_memory
                # Check if action_list is empty or doesn't match date_list length
                if len(action_list) == 0:
                    print("Warning: Empty action list in save_action_memory. Returning empty DataFrame.")
                    return pd.DataFrame()
                
                if len(action_list) != len(date_list):
                    print(f"Warning: Action list length ({len(action_list)}) doesn't match date list length ({len(date_list)})")
                    # Truncate to shorter length
                    min_len = min(len(action_list), len(date_list))
                    action_list = action_list[:min_len]
                    date_list = date_list[:min_len]
                    df_date = pd.DataFrame(date_list)
                    df_date.columns = ["date"]
                
                df_actions = pd.DataFrame(action_list)
                
                # Check if we have valid tic values
                if len(self.data.tic.values) == 0:
                    print("Warning: No tic values available. Using generic column names.")
                    df_actions.columns = [f"stock_{i}" for i in range(df_actions.shape[1])]
                else:
                    # Make sure column count matches
                    if df_actions.shape[1] != len(self.data.tic.values):
                        print(f"Warning: Action columns ({df_actions.shape[1]}) don't match tic count ({len(self.data.tic.values)})")
                        # Use generic column names
                        df_actions.columns = [f"stock_{i}" for i in range(df_actions.shape[1])]
                    else:
                        df_actions.columns = self.data.tic.values
                
                df_actions.index = df_date.date
            else:
                # For single stock
                date_list = self.date_memory[:-1]
                
                # Check if date_list is empty
                if len(date_list) == 0:
                    print("Warning: Empty date list in save_action_memory (single stock). Returning empty DataFrame.")
                    return pd.DataFrame()
                
                action_list = self.actions_memory
                
                # Check if action_list is empty or doesn't match date_list length
                if len(action_list) == 0:
                    print("Warning: Empty action list in save_action_memory (single stock). Returning empty DataFrame.")
                    return pd.DataFrame()
                
                if len(action_list) != len(date_list):
                    print(f"Warning: Action list length ({len(action_list)}) doesn't match date list length ({len(date_list)})")
                    # Truncate to shorter length
                    min_len = min(len(action_list), len(date_list))
                    action_list = action_list[:min_len]
                    date_list = date_list[:min_len]
                
                df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
            return df_actions
        except Exception as e:
            print(f"Error in save_action_memory: {e}")
            import traceback
            traceback.print_exc()
            # Return empty DataFrame as fallback
            return pd.DataFrame()

    def seed(self, seed=None):
        """Set the random seed"""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        """Get a vectorized environment for stable baselines"""
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
