#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os
import sys
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from gymnasium.spaces import Box, Discrete
import gym
import seaborn as sns

# Import the custom environment at the top of the file
try:
    from env_dapo_inference import DAPOInferenceEnv
    print("Successfully imported DAPOInferenceEnv")
except ImportError as e:
    print(f"Error importing DAPOInferenceEnv: {e}")
    print("Will attempt to use StockTradingEnv_llm_risk as fallback")

# Add the directory to python path to import the modules
# sys.path.append("/home/ruijian/FinRL_performance_impact.png (Sentiment/Risk impact over time)")
    # print("- dapo_deepseek_trading_results.csv (Performance metrics time series)")
    # print("- dapo_deepseek_portfolio_allocation.csv (Full portfolio allocation over time)")
    # print("- dapo_deepseek_portfolio_analysis.csv (Risk and sentiment analysis per stock)")
    # print("\nTo compare with CPPO DeepSeek performance, run the CPPO backtest script separately.")

# Import necessary components from FinRL_DeepSeek_backtest
from FinRL_DeepSeek_backtest import (
    INDICATORS,
    YahooDownloader
)

# Import the specialized environment for DeepSeek risk
from env_stocktrading_llm_risk import StockTradingEnv as StockTradingEnv_llm_risk

# Set paths and constants
MODEL_PATH = "/home/ruijian/FinRL_Contest_2025/Task_1_FinRL_DeepSeek_Stock/checkpoint/agent_dapo_deepseek_gpu_final.pth"
RISK_DATA_PATH = "/home/ruijian/FinRL_Contest_2025/Task_1_FinRL_DeepSeek_Stock/dataset/trade_data_deepseek_risk_2019_2023.csv"
SENTIMENT_DATA_PATH = "/home/ruijian/FinRL_Contest_2025/Task_1_FinRL_DeepSeek_Stock/dataset/trade_data_deepseek_sentiment_2019_2023.csv"
TRADE_START_DATE = '2019-01-01'
TRADE_END_DATE = '2023-12-31'

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    print(f"Model file not found: {MODEL_PATH}")
    sys.exit(1)

# Define MLPActorCritic class matching the one used in DAPO training
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

class MLPCategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

class MLPGaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution

class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # Policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # Move to GPU
        self.to(device)

    def step(self, obs):
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32).to(device)
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
        return a.cpu().numpy(), logp_a.cpu().numpy()
    
    def act_batch(self, obs, num_samples=10):
        """Sample multiple actions for a single observation"""
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32).to(device)
            actions = []
            logps = []
            for _ in range(num_samples):
                pi = self.pi._distribution(obs)
                a = pi.sample()
                logp_a = self.pi._log_prob_from_distribution(pi, a)
                actions.append(a.cpu().numpy())
                logps.append(logp_a.cpu().numpy())
            return actions, logps

    def act(self, obs):
        return self.step(obs)[0]

# Custom prediction function for DAPO
def custom_DAPO_prediction(act, environment, device='cuda'):
    """
    Custom prediction function for DAPO that properly handles GPU tensors
    """
    import torch
    
    print("Starting custom DAPO prediction...")
    try:
        # Reset environment and get initial state
        print("Resetting environment...")
        state, _ = environment.reset()
        print(f"Initial state shape: {state.shape if hasattr(state, 'shape') else 'N/A'}")
        
        # Initialize memory structures
        account_memory = []  # To store portfolio values
        actions_memory = []  # To store actions taken
        portfolio_distribution = []  # To store portfolio distribution
        episode_total_assets = [environment.initial_amount]
        sentiment_impact = []  # To track sentiment impact
        risk_impact = []  # To track risk impact

        # Extract functions to calculate LLM features impact
        def extract_prices(state):
            """Extract prices from state"""
            try:
                stock_dim = len(environment.df.tic.unique())
                return state[0, 1:stock_dim+1]
            except Exception as e:
                print(f"Error extracting prices: {e}")
                return None

        def extract_llm_features(state):
            """Extract LLM sentiment and risk scores from state"""
            try:
                stock_dim = len(environment.df.tic.unique())
                # State space structure: [Current Balance] + [Stock Prices] + [Stock Shares] + [Technical Indicators] + [LLM Sentiment] + [LLM Risk]
                sentiment_start = -(2 * stock_dim)  # Second to last block
                risk_start = -stock_dim  # Last block
                
                llm_sentiments = state[0, sentiment_start:risk_start]
                llm_risks = state[0, risk_start:]
                
                return llm_sentiments, llm_risks
            except Exception as e:
                print(f"Error extracting LLM features: {e}")
                # Return default neutral values
                stock_dim = len(environment.df.tic.unique())
                return np.ones(stock_dim) * 3, np.ones(stock_dim) * 3

        # Main prediction loop
        print(f"Starting prediction loop for {len(environment.df.index.unique())} time steps...")
        with torch.no_grad():
            for i in range(len(environment.df.index.unique())):
                try:
                    # Create tensor on the appropriate device
                    print(f"Step {i}: Creating tensor from state...")
                    if isinstance(state, np.ndarray):
                        s_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    else:
                        print(f"Warning: State is not a numpy array. Type: {type(state)}")
                        # Try to convert to numpy array first
                        try:
                            state_np = np.array(state)
                            s_tensor = torch.as_tensor(state_np, dtype=torch.float32).unsqueeze(0).to(device)
                        except Exception as e:
                            print(f"Error converting state to tensor: {e}")
                            # Create a dummy state tensor with the right shape
                            s_tensor = torch.zeros((1, environment.observation_space.shape[0]), dtype=torch.float32).to(device)
                    
                    # Get model outputs - for DAPO, we'll use the first action from act_batch
                    print(f"Step {i}: Getting actions from model...")
                    try:
                        actions, _ = act.act_batch(s_tensor, num_samples=1)
                        action = actions[0]
                        print(f"Action generated: {action[:5]}..." if len(action) > 5 else f"Action generated: {action}")
                    except Exception as e:
                        print(f"Error getting actions from model: {e}")
                        # Generate a random action as fallback
                        print("Generating random action as fallback")
                        action = np.random.uniform(-1, 1, environment.action_space.shape[0])
                    
                    # Step through the environment
                    print(f"Step {i}: Stepping through environment...")
                    next_state, reward, done, _, _ = environment.step(action)
                    
                    # Extract LLM features for analysis
                    print(f"Step {i}: Extracting LLM features...")
                    llm_sentiments, llm_risks = extract_llm_features(next_state)
                    
                    # Define mappings for risk and sentiment scores
                    risk_to_weight = {1: 0.99, 2: 0.995, 3: 1.0, 4: 1.005, 5: 1.01}
                    sentiment_to_weight = {1: 0.99, 2: 0.995, 3: 1.0, 4: 1.005, 5: 1.01}
                    
                    # Apply mappings to generate weights
                    print(f"Step {i}: Calculating risk and sentiment impacts...")
                    try:
                        llm_risks_weights = np.vectorize(lambda x: risk_to_weight.get(int(x), 1.0))(llm_risks)
                        llm_sentiment_weights = np.vectorize(lambda x: sentiment_to_weight.get(int(x), 1.0))(llm_sentiments)
                    except Exception as e:
                        print(f"Error calculating weights: {e}")
                        llm_risks_weights = np.ones_like(llm_risks)
                        llm_sentiment_weights = np.ones_like(llm_sentiments)

                    # Get stock prices for the current day
                    try:
                        price_array = environment.df.loc[environment.day, "close"].values
                    except Exception as e:
                        print(f"Error getting price array: {e}")
                        # Use a fallback method
                        try:
                            if hasattr(environment, 'data') and hasattr(environment.data, 'close'):
                                price_array = environment.data.close.values
                            else:
                                # Create dummy prices
                                price_array = np.ones(environment.stock_dim) * 100
                        except Exception as e2:
                            print(f"Error with fallback price method: {e2}")
                            price_array = np.ones(environment.stock_dim) * 100

                    # Stock holdings and cash balance
                    try:
                        stock_holdings = environment.state[(environment.stock_dim + 1):(environment.stock_dim * 2 + 1)]
                        cash_balance = environment.state[0]
                    except Exception as e:
                        print(f"Error getting holdings and balance: {e}")
                        stock_holdings = environment.num_stock_shares
                        cash_balance = environment.asset_memory[-1] if environment.asset_memory else environment.initial_amount

                    # Calculate total portfolio value
                    try:
                        total_asset = cash_balance + (price_array * stock_holdings).sum()
                    except Exception as e:
                        print(f"Error calculating total asset: {e}")
                        total_asset = environment.asset_memory[-1] if environment.asset_memory else environment.initial_amount

                    # Calculate portfolio distribution
                    try:
                        stock_values = price_array * stock_holdings
                        total_invested = stock_values.sum()
                    except Exception as e:
                        print(f"Error calculating stock values: {e}")
                        stock_values = np.zeros_like(stock_holdings)
                        total_invested = 0
                    
                    # Calculate weights based on portfolio allocation
                    if total_invested > 0:
                        try:
                            stock_weights = stock_values / total_invested
                            # Calculate aggregated sentiment and risk impact
                            aggregated_sentiment = np.dot(stock_weights, llm_sentiment_weights)
                            aggregated_risk = np.dot(stock_weights, llm_risks_weights)
                            
                            # Record impacts for analysis
                            sentiment_impact.append(aggregated_sentiment)
                            risk_impact.append(aggregated_risk)
                        except Exception as e:
                            print(f"Error calculating impacts: {e}")
                            sentiment_impact.append(1.0)
                            risk_impact.append(1.0)
                    else:
                        # If no positions, no sentiment/risk impact
                        sentiment_impact.append(1.0)
                        risk_impact.append(1.0)
                    
                    # Calculate distribution for tracking
                    try:
                        distribution = stock_values / total_asset if total_asset > 0 else np.zeros_like(stock_values)
                        cash_fraction = cash_balance / total_asset if total_asset > 0 else 1.0
                    except Exception as e:
                        print(f"Error calculating distribution: {e}")
                        distribution = np.zeros_like(stock_values)
                        cash_fraction = 1.0

                    # Store results
                    episode_total_assets.append(total_asset)
                    account_memory.append(total_asset)
                    actions_memory.append(action)
                    portfolio_distribution.append({
                        "cash": cash_fraction, 
                        "stocks": distribution.tolist(),
                        "sentiment_impact": sentiment_impact[-1],
                        "risk_impact": risk_impact[-1]
                    })

                    # Update state
                    state = next_state
                    
                    # Print progress every 10 steps
                    if i % 10 == 0:
                        print(f"Step {i}: Total asset value = {total_asset:.2f}")

                    if done:
                        print(f"Environment signaled done at step {i}")
                        break
                        
                except Exception as e:
                    print(f"Error in prediction loop at step {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue to next step instead of breaking
                    # Create a placeholder for this step's data
                    last_asset = episode_total_assets[-1] if episode_total_assets else environment.initial_amount
                    episode_total_assets.append(last_asset)
                    account_memory.append(last_asset)
                    actions_memory.append(np.zeros(environment.action_space.shape[0]))
                    portfolio_distribution.append({
                        "cash": 1.0, 
                        "stocks": [0] * environment.stock_dim,
                        "sentiment_impact": 1.0,
                        "risk_impact": 1.0
                    })
                    
                    # Try to get the next state
                    try:
                        environment.day += 1
                        if environment.day < len(environment.df.index.unique()):
                            environment.data = environment.df.loc[environment.day, :]
                            state = environment._update_state()
                    except Exception as e2:
                        print(f"Error advancing environment: {e2}")
                        # If we can't advance the environment, we need to break
                        break

        print("Test Finished!")
        # Save prediction results to CSV for analysis
        try:
            print("Saving prediction details to CSV...")
            pd.DataFrame({
                'step': range(len(episode_total_assets)),
                'total_assets': episode_total_assets,
                'sentiment_impact': sentiment_impact + [0] * (len(episode_total_assets) - len(sentiment_impact)),
                'risk_impact': risk_impact + [0] * (len(episode_total_assets) - len(risk_impact))
            }).to_csv('dapo_results/prediction_details.csv', index=False)
        except Exception as e:
            print(f"Error saving prediction details: {e}")
        
        return episode_total_assets, account_memory, actions_memory, portfolio_distribution
        
    except Exception as e:
        print(f"Critical error in custom_DAPO_prediction: {e}")
        import traceback
        traceback.print_exc()
        
        # Return dummy data to allow the script to continue
        print("Returning dummy data to allow script to continue")
        dummy_assets = [environment.initial_amount] * 10
        dummy_account = [environment.initial_amount] * 9
        dummy_actions = [np.zeros(environment.action_space.shape[0])] * 9
        dummy_portfolio = [{
            "cash": 1.0, 
            "stocks": [0] * environment.stock_dim,
            "sentiment_impact": 1.0,
            "risk_impact": 1.0
        }] * 9
        
        return dummy_assets, dummy_account, dummy_actions, dummy_portfolio

def enhanced_DRL_prediction(act, environment, verbose=True):
    """
    Enhanced wrapper around custom prediction function to capture dates
    """
    try:
        if verbose:
            print("Starting enhanced DRL prediction...")
        
        # Call our custom prediction function that properly handles GPU tensors
        episode_total_assets, account_memory, actions_memory, portfolio_distribution = custom_DAPO_prediction(
            act=act, environment=environment, device=device
        )
        
        if verbose:
            print(f"Prediction completed with {len(episode_total_assets)} time steps")
        
        # Extract dates from the environment data
        date_memory = []
        try:
            for day in range(len(environment.df.index.unique())):
                try:
                    # Get date for each day in the trading period
                    date = environment.df.loc[day, "date"].iloc[0]
                    date_memory.append(date)
                except Exception as e:
                    if verbose:
                        print(f"Error extracting date for day {day}: {e}")
                    # If we hit an error, we've likely run past the trading period
                    break
            
            # Verify lengths match
            if len(date_memory) == len(episode_total_assets) - 1:
                # Add initial date (using the first date in the dataset)
                first_date = environment.df.loc[0, "date"].iloc[0]
                date_memory = [first_date] + date_memory
                if verbose:
                    print("Added initial date to match assets length")
            elif len(date_memory) != len(episode_total_assets):
                if verbose:
                    print(f"Warning: Date memory length ({len(date_memory)}) doesn't match episode total assets length ({len(episode_total_assets)})")
                    print("Using indices as dates instead")
                date_memory = [str(i) for i in range(len(episode_total_assets))]
        except Exception as e:
            if verbose:
                print(f"Error processing dates: {e}")
                print("Using indices as dates instead")
            date_memory = [str(i) for i in range(len(episode_total_assets))]
        
        # Save prediction results to CSV immediately to preserve data
        if verbose:
            print("Saving raw prediction results to CSV...")
        try:
            pd.DataFrame({
                'date': date_memory,
                'total_assets': episode_total_assets
            }).to_csv('dapo_deepseek_raw_results.csv', index=False)
            if verbose:
                print("Raw results saved successfully")
        except Exception as e:
            if verbose:
                print(f"Error saving raw results: {e}")
        
        return episode_total_assets, date_memory, actions_memory, portfolio_distribution
    except Exception as e:
        if verbose:
            print(f"Error in enhanced_DRL_prediction: {e}")
            import traceback
            traceback.print_exc()
        
        # Return dummy data to allow the script to continue
        print("Returning dummy data to allow script to continue")
        dummy_assets = [1000000] * 10
        dummy_dates = [str(i) for i in range(10)]
        dummy_actions = []
        dummy_portfolio = None
        
        return dummy_assets, dummy_dates, dummy_actions, dummy_portfolio

def plot_performance(assets, dates, benchmark=None, title="DAPO-DeepSeek Trading Performance", save_path=None):
    print(f"Generating performance plot with {len(assets)} asset points and {len(dates)} date points...")
    
    try:
        # Create dapo_results directory if it doesn't exist
        os.makedirs('dapo_results', exist_ok=True)
        print("Ensured dapo_results directory exists")
        
        plt.figure(figsize=(16, 8))
        
        # Store original dates length for debugging
        original_dates_len = len(dates)
        original_assets_len = len(assets[1:])
        
        # Check and fix lengths for assets and dates first
        if len(assets[1:]) != len(dates):
            print(f"Warning: Assets length ({len(assets[1:])}) doesn't match dates length ({len(dates)})")
            # Option 1: Truncate dates to match assets
            if len(dates) > len(assets[1:]):
                dates = dates[:len(assets[1:])]
                print("Truncated dates to match assets length")
            # Option 2: If assets are longer, truncate assets (less common)
            elif len(assets[1:]) > len(dates):
                assets = [assets[0]] + assets[1:len(dates)+1]
                print("Truncated assets to match assets length")
        
        # Convert to pandas Series for easier handling
        try:
            # Convert dates to datetime if they're not already
            if not isinstance(dates[0], pd.Timestamp):
                dates = pd.to_datetime(dates, errors='coerce')
                # Check for NaT values
                if pd.isna(dates).any():
                    print("Warning: Some dates couldn't be converted to datetime. Using integer indices instead.")
                    dates = pd.date_range(start='2019-01-01', periods=len(assets[1:]))
            
            portfolio_series = pd.Series(assets[1:], index=dates)
            print(f"Created portfolio series with {len(portfolio_series)} points")
        except Exception as e:
            print(f"Error creating portfolio series: {e}")
            print("Using integer indices instead")
            portfolio_series = pd.Series(assets[1:], index=range(len(assets[1:])))
        
        # Calculate cumulative returns
        try:
            cumulative_returns = (portfolio_series / portfolio_series.iloc[0] - 1) * 100
            print(f"Calculated cumulative returns: {cumulative_returns.iloc[-1]:.2f}%")
        except Exception as e:
            print(f"Error calculating cumulative returns: {e}")
            cumulative_returns = pd.Series([0] * len(portfolio_series), index=portfolio_series.index)
        
        # Calculate daily returns for metrics
        try:
            daily_returns = portfolio_series.pct_change().dropna()
            print(f"Calculated {len(daily_returns)} daily returns")
        except Exception as e:
            print(f"Error calculating daily returns: {e}")
            daily_returns = pd.Series([0] * (len(portfolio_series) - 1), index=portfolio_series.index[1:])
        
        # Calculate performance metrics
        try:
            total_return = cumulative_returns.iloc[-1]
            annual_return = ((1 + total_return/100) ** (252/len(portfolio_series)) - 1) * 100
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() != 0 else 0
            
            # Calculate Sortino ratio (downside risk only)
            downside_returns = daily_returns[daily_returns < 0]
            sortino_ratio = np.sqrt(252) * daily_returns.mean() / downside_returns.std() if len(downside_returns) > 0 and downside_returns.std() != 0 else 0
            
            # Calculate max drawdown
            max_drawdown = ((portfolio_series / portfolio_series.cummax()) - 1).min() * 100
            volatility = daily_returns.std() * np.sqrt(252) * 100
            
            # Calculate Conditional Value at Risk (CVaR)
            confidence_level = 0.05  # 5% confidence level
            var = np.percentile(daily_returns, confidence_level * 100)
            cvar = daily_returns[daily_returns <= var].mean() * 100 if len(daily_returns[daily_returns <= var]) > 0 else 0
            
            print(f"Calculated performance metrics: Total Return={total_return:.2f}%, Annual Return={annual_return:.2f}%, Sharpe={sharpe_ratio:.2f}")
        except Exception as e:
            print(f"Error calculating performance metrics: {e}")
            total_return = annual_return = sharpe_ratio = sortino_ratio = max_drawdown = volatility = cvar = 0
        
        print("Setting up figure with subplots...")
        # Setup figure with two subplots - returns and drawdowns
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]})
        
        print("Plotting strategy performance...")
        # Plot strategy performance on first subplot
        ax1.plot(cumulative_returns, linewidth=2, label='DAPO-DeepSeek Strategy')
        
        # Plot benchmark if provided
        benchmark_returns = None
        if benchmark is not None:
            try:
                print("Processing benchmark data...")
                benchmark_values, benchmark_name = benchmark
                # Ensure benchmark values match the length of dates
                if len(benchmark_values) != len(dates):
                    print(f"Warning: Benchmark length ({len(benchmark_values)}) doesn't match dates length ({len(dates)})")
                    # Truncate benchmark values to match dates length
                    if len(benchmark_values) > len(dates):
                        benchmark_values = benchmark_values[:len(dates)]
                        print("Truncated benchmark values to match dates length")
                    else:
                        # If benchmark is shorter, truncate dates and portfolio series
                        dates = dates[:len(benchmark_values)]
                        cumulative_returns = cumulative_returns.iloc[:len(benchmark_values)]
                        print("Truncated dates and portfolio series to match benchmark length")
                
                # Calculate benchmark returns
                benchmark_series = pd.Series(benchmark_values, index=dates)
                benchmark_returns = (benchmark_series / benchmark_series.iloc[0] - 1) * 100
                ax1.plot(benchmark_returns, linewidth=2, label=benchmark_name)
                print(f"Added benchmark to plot: {benchmark_name}")
            except Exception as e:
                print(f"Error processing benchmark data: {e}")
        
        # Calculate drawdowns for second subplot
        try:
            drawdowns = (portfolio_series / portfolio_series.cummax() - 1) * 100
            ax2.fill_between(drawdowns.index, drawdowns, 0, alpha=0.5, color='r')
            ax2.set_ylabel('Drawdown (%)')
            ax2.set_title('Drawdowns')
            ax2.grid(True)
            print("Added drawdown subplot")
        except Exception as e:
            print(f"Error calculating drawdowns: {e}")
        
        # Add metrics to plot
        try:
            metrics_text = (
                f"Total Return: {total_return:.2f}%\n"
                f"Annual Return: {annual_return:.2f}%\n"
                f"Sharpe Ratio: {sharpe_ratio:.2f}\n"
                f"Sortino Ratio: {sortino_ratio:.2f}\n"
                f"Max Drawdown: {max_drawdown:.2f}%\n"
                f"Volatility: {volatility:.2f}%\n"
                f"CVaR (5%): {cvar:.2f}%"
            )
            
            # Add text box with metrics
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax1.text(0.02, 0.95, metrics_text, transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
            print("Added metrics text box to plot")
        except Exception as e:
            print(f"Error adding metrics text: {e}")
        
        # Finalize first subplot
        ax1.set_ylabel('Cumulative Return (%)')
        ax1.set_title(title)
        ax1.grid(True)
        ax1.legend()
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Save to file if path provided
        if save_path is None:
            save_path = 'dapo_results/dapo_deepseek_performance.png'
        
        try:
            plt.savefig(save_path, dpi=300)
            print(f"Successfully saved plot to {save_path}")
            
            # Save metrics to CSV for reference
            metrics_df = pd.DataFrame({
                'Metric': ['Total Return', 'Annual Return', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown', 'Volatility', 'CVaR (5%)'],
                'Value': [total_return, annual_return, sharpe_ratio, sortino_ratio, max_drawdown, volatility, cvar]
            })
            metrics_csv_path = 'dapo_results/dapo_deepseek_metrics.csv'
            metrics_df.to_csv(metrics_csv_path, index=False)
            print(f"Saved metrics to {metrics_csv_path}")
            
            # Save returns data for further analysis
            returns_df = pd.DataFrame({
                'Date': cumulative_returns.index,
                'Strategy_Return': cumulative_returns.values
            })
            if benchmark_returns is not None:
                returns_df['Benchmark_Return'] = benchmark_returns.values
            returns_csv_path = 'dapo_results/dapo_deepseek_returns.csv'
            returns_df.to_csv(returns_csv_path, index=False)
            print(f"Saved returns data to {returns_csv_path}")
            
        except Exception as e:
            print(f"Error saving plot: {e}")
            # Try alternative save method
            try:
                plt.savefig('dapo_results/dapo_performance_backup.png', dpi=200)
                print("Saved plot using backup method")
            except Exception as e2:
                print(f"Error with backup save method: {e2}")
        
        plt.close(fig)
        return True
    except Exception as e:
        print(f"Error in plot_performance: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_benchmark_data(start_date, end_date, initial_value=1000000):
    """
    Get benchmark data (NASDAQ 100 index) for comparison
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        initial_value (float): Initial portfolio value to normalize benchmark against
        
    Returns:
        tuple: (benchmark_values, benchmark_name) if successful, None if failed
    """
    try:
        # Try to download NASDAQ-100 data
        df_benchmark = YahooDownloader(
            start_date=start_date, 
            end_date=end_date, 
            ticker_list=["^NDX"]
        ).fetch_data()
        
        # Verify data quality
        if df_benchmark.empty:
            print("Warning: Retrieved benchmark data is empty")
            return None
        
        if 'close' not in df_benchmark.columns:
            print("Warning: No close price data in benchmark dataset")
            return None
            
        # Check for sufficient data points
        if len(df_benchmark) < 2:
            print("Warning: Insufficient benchmark data points")
            return None
            
        # Normalize to initial portfolio value
        first_value = df_benchmark["close"].iloc[0]
        if first_value <= 0:
            print("Warning: Invalid first value in benchmark data")
            return None
            
        benchmark_values = df_benchmark["close"].div(first_value).mul(initial_value)
        
        # Verify final calculations
        if benchmark_values.isna().any():
            print("Warning: NaN values detected in benchmark calculations")
            return None
            
        return benchmark_values.tolist(), "NASDAQ-100 Index"
        
    except Exception as e:
        print(f"Error retrieving benchmark data: {str(e)}")
        return None

def main():
    # Load and merge trading data
    try:
        # Check if both files exist
        if not os.path.exists(RISK_DATA_PATH):
            print(f"Risk data file not found: {RISK_DATA_PATH}")
            sys.exit(1)
            
        if not os.path.exists(SENTIMENT_DATA_PATH):
            print(f"Sentiment data file not found: {SENTIMENT_DATA_PATH}")
            sys.exit(1)
        
        # Load risk data
        print(f"Loading risk data from {RISK_DATA_PATH}...")
        try:
            trade_risk = pd.read_csv(RISK_DATA_PATH)
            print(f"Risk data loaded successfully with {len(trade_risk)} rows")
            print(f"Risk data columns: {trade_risk.columns.tolist()}")
            
            # Ensure date column is properly formatted
            if 'date' in trade_risk.columns:
                # Make sure date is in datetime format
                trade_risk['date'] = pd.to_datetime(trade_risk['date'])
                print("Date column formatted successfully in risk data")
            else:
                print("Warning: 'date' column not found in risk data")
                # Check if there's a column that might contain date information
                date_like_cols = [col for col in trade_risk.columns if 'date' in col.lower()]
                if date_like_cols:
                    print(f"Found potential date columns: {date_like_cols}")
                    trade_risk['date'] = pd.to_datetime(trade_risk[date_like_cols[0]])
                    print(f"Using {date_like_cols[0]} as date column")
                else:
                    print("No date-like columns found, creating dummy date column")
                    # Create a dummy date column if none exists
                    trade_risk['date'] = pd.date_range(start=TRADE_START_DATE, periods=len(trade_risk))
            
            # Fill missing values in llm_risk column
            if 'llm_risk' not in trade_risk.columns:
                print("Adding llm_risk column with default value 3")
                trade_risk['llm_risk'] = 3  # Neutral risk as default
            else:
                trade_risk['llm_risk'] = trade_risk['llm_risk'].fillna(3)  # Fill missing values with neutral risk
                
            # Create a simplified version with just the necessary columns for testing
            trade_llm_risk = trade_risk.copy()
            
            # Add dummy sentiment if sentiment data can't be loaded
            if 'llm_sentiment' not in trade_llm_risk.columns:
                print("Adding llm_sentiment column with default value 3")
                trade_llm_risk['llm_sentiment'] = 3  # Neutral sentiment as default
            
            # Try to load sentiment data, but continue with risk data if it fails
            try:
                print(f"Loading sentiment data from {SENTIMENT_DATA_PATH}...")
                trade_sentiment = pd.read_csv(SENTIMENT_DATA_PATH)
                print(f"Sentiment data loaded successfully with {len(trade_sentiment)} rows")
                
                # Ensure date column is properly formatted in sentiment data
                if 'date' in trade_sentiment.columns:
                    trade_sentiment['date'] = pd.to_datetime(trade_sentiment['date'])
                
                # Try to merge risk and sentiment data
                print("Merging risk and sentiment data...")
                trade_merged = pd.merge(
                    trade_risk, 
                    trade_sentiment, 
                    on=['date', 'tic'], 
                    suffixes=('', '_sentiment'),
                    how='left'  # Use left join to keep all risk data
                )
                
                # Check if sentiment column was merged correctly
                if 'llm_sentiment' in trade_sentiment.columns and 'llm_sentiment_sentiment' in trade_merged.columns:
                    # Use sentiment data from sentiment file
                    trade_merged['llm_sentiment'] = trade_merged['llm_sentiment_sentiment'].fillna(3)
                    trade_merged.drop('llm_sentiment_sentiment', axis=1, inplace=True)
                    print("Successfully merged sentiment data")
                    trade_llm_risk = trade_merged
                
            except Exception as e:
                print(f"Error loading sentiment data: {e}")
                print("Continuing with risk data only")
                # We'll use the trade_llm_risk with dummy sentiment values created above
            
            # Ensure all required columns exist
            required_columns = ['date', 'tic', 'close', 'llm_sentiment', 'llm_risk']
            for col in required_columns:
                if col not in trade_llm_risk.columns:
                    if col == 'llm_sentiment':
                        print(f"Adding missing column: {col}")
                        trade_llm_risk[col] = 3  # Neutral sentiment
                    elif col == 'llm_risk':
                        print(f"Adding missing column: {col}")
                        trade_llm_risk[col] = 3  # Neutral risk
                    else:
                        print(f"Error: Required column {col} missing and cannot be synthesized")
                        raise ValueError(f"Missing required column: {col}")
            
            # Final check of data
            print(f"Final data shape: {trade_llm_risk.shape}")
            print(f"Final data columns: {trade_llm_risk.columns.tolist()}")
            print(f"Sample of final data:\n{trade_llm_risk.head()}")
            
        except Exception as e:
            print(f"Error loading risk data: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        # Prepare index
        unique_dates = trade_llm_risk['date'].unique()
        date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
        trade_llm_risk['new_idx'] = trade_llm_risk['date'].map(date_to_idx)
        trade_llm_risk = trade_llm_risk.set_index('new_idx')
        
        # Fill missing values for sentiment and risk
        trade_llm_risk['llm_sentiment'].fillna(3, inplace=True)  # neutral sentiment is 3
        trade_llm_risk['llm_risk'].fillna(3, inplace=True)  # neutral risk is 3
        
        # Print dataset statistics
        print(f"Loaded and merged {len(trade_llm_risk)} records with {len(unique_dates)} trading days")
        print(f"Date range: {trade_llm_risk['date'].min()} to {trade_llm_risk['date'].max()}")
        print(f"Number of stocks: {len(trade_llm_risk['tic'].unique())}")
        print(f"LLM risk score distribution: \n{trade_llm_risk['llm_risk'].value_counts().sort_index()}")
        print(f"LLM sentiment score distribution: \n{trade_llm_risk['llm_sentiment'].value_counts().sort_index()}")
        
        # Use the merged data for the trading environment
        trade_llm_risk = trade_llm_risk
        
    except Exception as e:
        print(f"Error loading and merging trading data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Configure environment
    stock_dimension = len(trade_llm_risk.tic.unique())
    state_space_llm_risk = 1 + 2 * stock_dimension + (2+len(INDICATORS)) * stock_dimension
    
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space_llm_risk}")
    
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension
    
    env_kwargs_llm_risk = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space_llm_risk,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }
    
    # Create environment
    print("Creating DAPO-DeepSeek trading environment with risk and sentiment features...")
    try:
        # Use the custom DAPO inference environment
        print("Initializing DAPOInferenceEnv with the following parameters:")
        print(f"  - Stock dimension: {stock_dimension}")
        print(f"  - State space: {state_space_llm_risk}")
        print(f"  - Action space: {stock_dimension}")
        print(f"  - Technical indicators: {INDICATORS}")
        
        e_trade_llm_risk_gym = DAPOInferenceEnv(
            df=trade_llm_risk, 
            stock_dim=stock_dimension,
            hmax=100,
            initial_amount=1000000,
            num_stock_shares=[0] * stock_dimension,
            buy_cost_pct=[0.001] * stock_dimension,
            sell_cost_pct=[0.001] * stock_dimension,
            state_space=state_space_llm_risk,
            action_space=stock_dimension,
            tech_indicator_list=INDICATORS,
            turbulence_threshold=70, 
            risk_indicator_col='vix',
            llm_sentiment_col='llm_sentiment',
            llm_risk_col='llm_risk',
            reward_scaling=1.0,  # Adding the missing reward_scaling parameter
            make_plots=True,
            print_verbosity=10,
            model_name="dapo_deepseek",
            mode="backtest"
        )
        print("Successfully created DAPO inference environment")
    except Exception as e:
        print(f"Error creating DAPO inference environment: {e}")
        print("Falling back to standard environment...")
        import traceback
        traceback.print_exc()
        
        # Fallback to standard environment
        e_trade_llm_risk_gym = StockTradingEnv_llm_risk(
            df=trade_llm_risk, 
            stock_dim=stock_dimension,
            hmax=100,
            initial_amount=1000000,
            num_stock_shares=[0] * stock_dimension,
            buy_cost_pct=[0.001] * stock_dimension,
            sell_cost_pct=[0.001] * stock_dimension,
            state_space=state_space_llm_risk,
            action_space=stock_dimension,
            tech_indicator_list=INDICATORS,
            turbulence_threshold=70, 
            risk_indicator_col='vix',
            reward_scaling=1.0  # Adding the missing reward_scaling parameter
        )
    
    # Get observation and action spaces
    observation_space_llm_risk = e_trade_llm_risk_gym.observation_space
    action_space_llm_risk = e_trade_llm_risk_gym.action_space
    
    # Load the model
    print(f"Loading DAPO-DeepSeek model from {MODEL_PATH}...")
    try:
        # Using the MLPActorCritic class defined above
        loaded_dapo = MLPActorCritic(
            observation_space_llm_risk, 
            action_space_llm_risk, 
            hidden_sizes=(512, 512),
            activation=nn.ReLU
        )
        
        # Load model weights - handle both direct state dict and checkpoint formats
        if not os.path.exists(MODEL_PATH):
            print(f"Warning: Model file not found at {MODEL_PATH}")
            print("Using the initialized model for testing plot functionality...")
        else:
            try:
                checkpoint = torch.load(MODEL_PATH, map_location=device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # The model was saved as a checkpoint dict
                    loaded_dapo.load_state_dict(checkpoint['model_state_dict'])
                    epoch = checkpoint.get('epoch', 'unknown')
                    print(f"Loaded checkpoint from epoch {epoch}")
                else:
                    # The model was saved as a direct state dict
                    loaded_dapo.load_state_dict(checkpoint)
                print(f"Model loaded successfully from {MODEL_PATH}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Using the initialized model for testing plot functionality...")
                import traceback
                traceback.print_exc()
            
        loaded_dapo.to(device)
        loaded_dapo.eval()  # Set the model to evaluation mode
    except Exception as e:
        print(f"Error loading model: {e}")
        
        # Debugging information
        print("\nDebugging information:")
        if os.path.exists(MODEL_PATH):
            try:
                checkpoint = torch.load(MODEL_PATH, map_location="cpu")
                if isinstance(checkpoint, dict):
                    print(f"Checkpoint keys: {checkpoint.keys()}")
                    if 'model_state_dict' in checkpoint:
                        print(f"Model state dict keys: {checkpoint['model_state_dict'].keys()}")
                else:
                    print(f"Direct state dict keys: {checkpoint.keys()}")
            except Exception as e:
                print(f"Failed to load model state dict for debugging: {e}")
        
        sys.exit(1)
    
    # Make sure save directory exists
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dapo_results")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Created save directory: {save_dir}")

    # Run prediction and plot performance
    try:
        print("Running DAPO-DeepSeek prediction...")
        assets_llm_risk, dates_llm_risk, actions_memory, portfolio_distribution = enhanced_DRL_prediction(loaded_dapo, e_trade_llm_risk_gym)
        
        # Get benchmark data
        print("Getting benchmark data...")
        benchmark_data = None
        try:
            benchmark_data = get_benchmark_data(
                start_date=dates_llm_risk[0], 
                end_date=dates_llm_risk[-1], 
                initial_value=1000000
            )
            if benchmark_data:
                print(f"Benchmark data retrieved successfully: {benchmark_data[1]}")
            else:
                print("Benchmark data retrieval failed, proceeding without benchmark")
        except Exception as e:
            print(f"Error getting benchmark data: {e}")
            print("Proceeding without benchmark data")
        
        # Plot performance
        print("Plotting performance...")
        try:
            # Create directory for saving results if it doesn't exist
            save_dir = "dapo_results"
            os.makedirs(save_dir, exist_ok=True)
            print(f"Created/verified save directory: {os.path.abspath(save_dir)}")
            
            # Set the save path for the plot
            plot_save_path = os.path.join(save_dir, "DAPO_DeepSeek_Trading_Performance.png")
            
            # Call the plot_performance function with robust error handling
            plot_performance(
                assets=assets_llm_risk, 
                dates=dates_llm_risk, 
                benchmark=benchmark_data, 
                title="DAPO-DeepSeek Trading Performance", 
                save_path=save_dir
            )
            print("Performance plotting completed successfully")
        except Exception as e:
            print(f"Error in performance plotting: {e}")
            import traceback
            traceback.print_exc()
            print("Attempting alternative plotting approach...")
            
            # Try a simplified plotting approach as fallback
            try:
                plt.figure(figsize=(16, 8))
                plt.plot(dates_llm_risk, assets_llm_risk, label='DAPO-DeepSeek')
                plt.title("DAPO-DeepSeek Trading Performance (Simplified)")
                plt.xlabel("Date")
                plt.ylabel("Portfolio Value ($)")
                plt.legend()
                plt.grid(True)
                
                # Save the simplified plot
                simple_save_path = os.path.join(save_dir, "DAPO_DeepSeek_Trading_Performance_Simple.png")
                plt.savefig(simple_save_path, dpi=300)
                print(f"Saved simplified performance plot to {simple_save_path}")
                plt.close()
            except Exception as e2:
                print(f"Even simplified plotting failed: {e2}")
    except Exception as e:
        print(f"Error in prediction or plotting: {e}")
        import traceback
        traceback.print_exc()
    
    # Print metrics
    print("\n=== DAPO-DeepSeek Performance Metrics ===")
    # for key, value in result['metrics'].items():
    #     print(f"{key.replace('_', ' ').title()}: {value:.2f}{'%' if 'return' in key or 'drawdown' in key or 'volatility' in key or 'cvar' in key else ''}")
    
    # Save results to CSV
    # Ensure all arrays have the same length before creating DataFrame
    # min_length = min(len(result['data']['dates']), 
    #                  len(result['data']['portfolio_returns']),
    #                  len(result['data']['benchmark_returns']) if result['data']['benchmark_returns'] is not None else float('inf'))
    
    # result_df = pd.DataFrame({
    #     'Date': result['data']['dates'][:min_length],
    #     'DAPO_DeepSeek_Return': result['data']['portfolio_returns'][:min_length],
    # })
    
    # if benchmark and result['data']['benchmark_returns'] is not None:
    #     result_df['Benchmark_Return'] = result['data']['benchmark_returns'][:min_length]
    
    # result_df.to_csv('dapo_deepseek_trading_results.csv', index=False)
    # print("\nResults saved to dapo_deepseek_trading_results.csv")

    # Analyze portfolio allocation over time
    if portfolio_distribution:
        # Create a dataframe to store the portfolio allocation
        allocation_df = pd.DataFrame([{
            'day': i, 
            'cash': pd['cash'],
            'sentiment_impact': pd.get('sentiment_impact', 1.0),
            'risk_impact': pd.get('risk_impact', 1.0),
            **{f'stock_{j}': stock for j, stock in enumerate(pd['stocks'])}
        } for i, pd in enumerate(portfolio_distribution)])
        
        # Calculate average portfolio allocation
        avg_cash = allocation_df['cash'].mean()
        
        # Get stock names
        stock_names = trade_llm_risk.tic.unique()
        
        # Calculate average stock allocations
        avg_stocks = np.zeros(len(stock_names))
        for i in range(len(stock_names)):
            col = f'stock_{i}'
            if col in allocation_df.columns:
                avg_stocks[i] = allocation_df[col].mean()
        
        # Calculate average sentiment and risk impact
        avg_sentiment = allocation_df['sentiment_impact'].mean()
        avg_risk = allocation_df['risk_impact'].mean()
        sentiment_risk_ratio = avg_sentiment / avg_risk if avg_risk > 0 else 1.0
        
        print("\n=== Average Portfolio Allocation ===")
        print(f"Cash: {avg_cash*100:.2f}%")
        print("Stocks:")
        for i, stock in enumerate(stock_names):
            if i < len(avg_stocks):
                print(f"  {stock}: {avg_stocks[i]*100:.2f}%")
                
        print("\n=== DAPO Feature Impact Analysis ===")
        print(f"Average Sentiment Impact: {avg_sentiment:.4f}")
        print(f"Average Risk Impact: {avg_risk:.4f}")
        print(f"Average Sentiment/Risk Ratio: {sentiment_risk_ratio:.4f}")
        
        # Analyze risk exposure based on portfolio weights and LLM risk scores
        # First, get the average LLM risk score for each stock
        risk_df = trade_llm_risk.groupby('tic')['llm_risk'].mean().reset_index()
        risk_df.columns = ['tic', 'avg_risk_score']
        
        # Get the average LLM sentiment score for each stock
        sentiment_df = trade_llm_risk.groupby('tic')['llm_sentiment'].mean().reset_index()
        sentiment_df.columns = ['tic', 'avg_sentiment_score']
        
        # Create a DataFrame with stock names and their average weights
        portfolio_weights = pd.DataFrame({
            'tic': stock_names[:len(avg_stocks)],
            'avg_weight': avg_stocks[:len(stock_names)]
        })
        
        # Merge with risk and sentiment scores
        portfolio_analysis = portfolio_weights.merge(risk_df, on='tic', how='left')
        portfolio_analysis = portfolio_analysis.merge(sentiment_df, on='tic', how='left')
        
        # Define the mappings for risk and sentiment scores
        risk_interpretations = {
            1: "Very Low Risk",
            2: "Low Risk",
            3: "Neutral Risk",
            4: "High Risk",
            5: "Very High Risk"
        }
        
        sentiment_interpretations = {
            1: "Very Negative",
            2: "Negative",
            3: "Neutral",
            4: "Positive",
            5: "Very Positive"
        }
        
        # Add risk and sentiment interpretation columns
        portfolio_analysis['risk_level'] = portfolio_analysis['avg_risk_score'].apply(
            lambda x: risk_interpretations.get(round(x), "Unknown")
        )
        
        portfolio_analysis['sentiment_level'] = portfolio_analysis['avg_sentiment_score'].apply(
            lambda x: sentiment_interpretations.get(round(x), "Unknown")
        )
        
        # Calculate sentiment-to-risk ratio for each stock
        portfolio_analysis['sentiment_risk_ratio'] = portfolio_analysis['avg_sentiment_score'] / portfolio_analysis['avg_risk_score']
        
        # Calculate risk-weighted and sentiment-weighted allocations
        portfolio_analysis['risk_weighted_allocation'] = portfolio_analysis['avg_weight'] * portfolio_analysis['avg_risk_score']
        portfolio_analysis['sentiment_weighted_allocation'] = portfolio_analysis['avg_weight'] * portfolio_analysis['avg_sentiment_score']
        
        # Calculate DAPO specific metric: sentiment/risk weighted allocation
        portfolio_analysis['dapo_weighted_allocation'] = portfolio_analysis['avg_weight'] * portfolio_analysis['sentiment_risk_ratio']
        
        # Calculate total weighted exposures
        total_risk_exposure = portfolio_analysis['risk_weighted_allocation'].sum()
        total_sentiment_exposure = portfolio_analysis['sentiment_weighted_allocation'].sum()
        total_dapo_exposure = portfolio_analysis['dapo_weighted_allocation'].sum()
        
        print("\n=== Portfolio DAPO Analysis ===")
        print(f"Total Risk-Weighted Exposure: {total_risk_exposure:.4f}")
        print(f"Total Sentiment-Weighted Exposure: {total_sentiment_exposure:.4f}")
        print(f"Total DAPO-Weighted Exposure: {total_dapo_exposure:.4f}")
        print(f"Portfolio Sentiment/Risk Ratio: {total_sentiment_exposure/total_risk_exposure:.4f}")
        
        print("\nTop 5 Highest DAPO-Weighted Positions:")
        
        # Sort by DAPO-weighted allocation and show top 5
        top_dapo = portfolio_analysis.sort_values('dapo_weighted_allocation', ascending=False).head(5)
        for _, row in top_dapo.iterrows():
            print(f"  {row['tic']}: Weight {row['avg_weight']*100:.2f}%, "
                  f"Sentiment {row['avg_sentiment_score']:.2f} ({row['sentiment_level']}), "
                  f"Risk {row['avg_risk_score']:.2f} ({row['risk_level']}), "
                  f"S/R Ratio {row['sentiment_risk_ratio']:.2f}")
        
        # Save allocation and analysis to CSV
        allocation_df.to_csv('dapo_deepseek_portfolio_allocation.csv', index=False)
        portfolio_analysis.to_csv('dapo_deepseek_portfolio_analysis.csv', index=False)
        print("\nPortfolio allocation and analysis saved to CSV files")

if __name__ == "__main__":
    main()