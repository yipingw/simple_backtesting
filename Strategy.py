from utils import assert_msg, crossover, SMA
import abc
import numpy as np
from typing import Callable


class Strategy(metaclass=abc.ABCMeta):
    """
    Abstract strategy class for defining trading strategies.

    To define your own strategy class, inherit from this base class and implement two abstract methods:
    Strategy.init
    Strategy.next
    """

    def __init__(self, broker, data):
        """
        Construct a strategy object.

        @params broker:  ExchangeAPI    Trading API interface for simulating trades
        @params data:    list           Market data
        """
        self._indicators = []
        self._broker = broker
        self._data = data
        self._tick = 0

    def I(self, func: Callable, *args) -> np.ndarray:
        """
        Calculate buy/sell indicator vectors. Buy/sell indicator vectors are arrays with a length corresponding to the historical data;
        used to determine whether to "buy" or "sell" at this point in time.

        For example, calculating a moving average:
        def init():
            self.sma = self.I(utils.SMA, self.data.Close, N)
        """
        value = func(*args)
        value = np.asarray(value)
        assert_msg(value.shape[-1] == len(self._data.Close),
                   'Indicator length must match data length')

        self._indicators.append(value)
        return value

    @property
    def tick(self):
        return self._tick

    @abc.abstractmethod
    def init(self):
        """
        Initialize the strategy. Called once during the strategy backtesting/execution process to initialize internal state of the strategy.
        Auxiliary parameters for the strategy can also be precomputed here. For example, based on historical market data:
        Calculate buy/sell indicator vectors;
        Train models/initialize model parameters
        """
        pass

    @abc.abstractmethod
    def next(self, tick):
        """
        Step function, execute the strategy at tick step. 'tick' represents the current "time". For example, data[tick] is used to access the current market price.
        """
        pass

    def buy(self):
        self._broker.buy()

    def sell(self):
        self._broker.sell()

    @property
    def data(self):
        return self._data


class SmaCross(Strategy):
    # Window size for the fast SMA, used to calculate the SMA fast line
    fast = 30

    # Window size for the slow SMA, used to calculate the SMA slow line
    slow = 90

    def init(self):
        # Calculate the fast and slow lines at each moment in history
        self.sma1 = self.I(SMA, self.data.Close, self.fast)
        self.sma2 = self.I(SMA, self.data.Close, self.slow)

    def next(self, tick):
        # If at this moment the fast line just crosses above the slow line, buy everything
        if crossover(self.sma1[:tick], self.sma2[:tick]):
            self.buy()

        # If the slow line just crosses above the fast line, sell everything
        elif crossover(self.sma2[:tick], self.sma1[:tick]):
            self.sell()

        # Otherwise, do not perform any actions at this moment.
        else:
            pass
