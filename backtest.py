import numpy as np
import pandas as pd
from numbers import Number

from Strategy import Strategy, SmaCross
from utils import read_file, assert_msg, crossover, SMA


class ExchangeAPI:
    def __init__(self, data, cash, commission):
        assert_msg(0 < cash, "Initial cash amount must be greater than 0, the input cash amount: {}".format(cash))
        assert_msg(0 <= commission <= 0.05, "A reasonable commission rate generally does not exceed 5%, the input rate: {}".format(commission))
        self._inital_cash = cash
        self._data = data
        self._commission = commission
        self._position = 0
        self._cash = cash
        self._i = 0

    @property
    def cash(self):
        """
        :return: Returns the current account cash amount
        """
        return self._cash

    @property
    def position(self):
        """
        :return: Returns the current account position
        """
        return self._position

    @property
    def initial_cash(self):
        """
        :return: Returns the initial cash amount
        """
        return self._inital_cash

    @property
    def market_value(self):
        """
        :return: Returns the current market value
        """
        return self._cash + self._position * self.current_price

    @property
    def current_price(self):
        """
        :return: Returns the current market price
        """
        return self._data.Close[self._i]

    def buy(self):
        """
        Buy with the remaining funds in the current account at market price
        """
        self._position = float(self._cash * (1 - self._commission) / self.current_price)
        self._cash = 0.0

    def sell(self):
        """
        Sell the remaining position in the current account
        """
        self._cash += float(self._position * self.current_price * (1 - self._commission))
        self._position = 0.0

    def next(self, tick):
        self._i = tick


class Backtest:
    """
    Backtest class, used for reading historical market data, executing strategies, simulating trades, and estimating
    returns.

    At initialization, call Backtest.run to backtest

    instance, or `backtesting.backtesting.Backtest.optimize` to
    optimize it.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 strategy_type: type(Strategy),
                 broker_type: type(ExchangeAPI),
                 cash: float = 10000,
                 commission: float = .0):
        """
        Construct the backtest object. Required parameters include: historical data, strategy object, initial cash amount, commission rate, etc.
        The initialization process includes checking input types, filling in data null values, etc.

        Parameters:
        :param data:            pd.DataFrame        pandas Dataframe format of historical OHLCV data
        :param broker_type:     type(ExchangeAPI)   Exchange API type, responsible for executing buy/sell operations and maintaining account status
        :param strategy_type:   type(Strategy)      Strategy type
        :param cash:            float               Initial cash amount
        :param commission:      float               Transaction commission rate per trade. E.g., for 2% commission, this would be 0.02
        """

        assert_msg(issubclass(strategy_type, Strategy), 'strategy_type is not a Strategy type')
        assert_msg(issubclass(broker_type, ExchangeAPI), 'strategy_type is not a Strategy type')
        assert_msg(isinstance(commission, Number), 'commission is not a floating-point number type')

        data = data.copy(False)

        # Fill NaN if there's no Volume column
        if 'Volume' not in data:
            data['Volume'] = np.nan

        # Validate OHLC data format
        assert_msg(len(data.columns & {'Open', 'High', 'Low', 'Close', 'Volume'}) == 5,
                   "The input `data` format is incorrect, it must include these columns at minimum: "
                   "'Open', 'High', 'Low', 'Close'")

        # Check for missing values
        assert_msg(not data[['Open', 'High', 'Low', 'Close']].max().isnull().any(),
            "Some OHLC contains missing values, please remove those rows or fill them with interpolation.")

        # Re-sort if market data is not in chronological order
        if not data.index.is_monotonic_increasing:
            data = data.sort_index()

        # Initialize exchange object and strategy object using data.
        self._data = data  # type: pd.DataFrame
        self._broker = broker_type(data, cash, commission)
        self._strategy = strategy_type(self._broker, self._data)
        self._results = None

    def run(self) -> pd.Series:
        """
        Run the backtest, iterate over historical data, execute simulated trades and return backtest results.
        """
        strategy = self._strategy
        broker = self._broker

        # Strategy initialization
        strategy.init()

        # Set start and end for backtesting
        start = 100
        end = len(self._data)

        # Main backtesting loop, update market status, then execute strategy
        for i in range(start, end):
            # Note to move market status to the ith moment first, then execute the strategy.
            broker.next(i)
            strategy.next(i)

        # After finishing strategy execution, calculate results and return
        self._results = self._compute_result(broker)
        return self._results

    def _compute_result(self, broker):
        s = pd.Series()
        s['Initial market value'] = broker.initial_cash
        s['Final market value'] = broker.market_value
        s['Profit'] = broker.market_value - broker.initial_cash
        return s


def main():
    BTCUSD = read_file('BTCUSD_GEMINI.csv')
    ret = Backtest(BTCUSD, SmaCross, ExchangeAPI, 10000.0, 0.003).run()
    print(ret)

if __name__ == '__main__':
    main()
