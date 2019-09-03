from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import backtrader as bt
import backtrader.feeds as btfeeds
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import numpy as np
from tabulate import tabulate
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
import random
from copy import deepcopy
import math

# Analyzer Optimization
init_wf = False
monte_test = False
multi_data = False
single_data = True



# Strategy to be intialized
class TestStrategy(bt.Strategy):

    params = (('period', 20),
              ('printlog', True),
              ('ex_rate', 0.70),
              ('counter_currency', True),
              ('base_currency', False),
              ('JPY_pair', False),
              ('no_currency', False),
              ('pct_risk', 0.02),
              ('express_exec',False),
              ('EOE', True),
              ('oneplot',False),
              ('order_market',bt.Order.Market),
              ('order_limit', bt.Order.Limit),
              ('order_close', bt.Order.Stop),
              )

    # LOG METHOD
    def log(self, txt, dt=None, dprint=True):
        if self.params.printlog or dprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    # Values to load first
    def __init__(self):

        #MULTIS DATA FEED
        if multi_data:
            self.inds = dict()
            for i, d in enumerate(self.datas):

                self.muldata = d
                self.units = 0
                self.inds[self.muldata] = dict()
                self.inds[self.muldata]['atr'] = bt.indicators.ATR(self.muldata, period = 14)
                self.inds[self.muldata]['sma1'] = bt.indicators.SimpleMovingAverage(self.muldata.close, period = 40)
                self.inds[self.muldata]['sma2'] = bt.indicators.SimpleMovingAverage(self.muldata.close, period = 100)
                self.inds[self.muldata]['cross'] = bt.indicators.CrossOver(self.inds[self.muldata]['sma1'], self.inds[self.muldata]['sma2'])
                if i > 0:
                    if self.params.oneplot:
                        d.plotinfo.plotmaster = self.datas[0]

        #SINGLE DATA FEED                
        else:
            # INIT DATA VALUES-------------
            self.dataclose = self.datas[0].close
            self.dataopen = self.datas[0].open
            self.datahigh = self.datas[0].high
            self.datalow = self.datas[0].low
            self.datavolume = self.datas[0].volume
            self.units = 0
            # INIT INDICATORS--------------
            #self.rsi = bt.indicators.RSI_SMA(self.datas[0], period=self.params.period)
            self.sma = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.period)
            self.order = None
            self.volume = self.datas[0].volume
            self.atr = bt.indicators.ATR(self.datas[0], period=14)
            self.macdInd = bt.indicators.MACD(self.datas[0], period_me1=12, period_me2=26, period_signal=9)
            # self.bbands = bt.indicators.BBands(self.datas[0], period=self.params.period)
            # self.ichi = bt.indicators.Ichimoku(self.datas[0], tenkan=9, kijun=26, senkou=52, senkou_lead=26, chikou=26)
            self.aroon = bt.indicators.AroonIndicator(self.datas[0], period=self.p.period)
            self.trade_sig = bt.indicators.CrossOver(self.aroon.lines.aroonup, self.aroon.lines.aroondown)

    # LOG ORDERS--------------
    def notify_order(self, order):
        date = self.data.datetime.datetime().date()
        ep = order.executed.price
        es = order.executed.size
        tv = ep * es
        leverage = self.broker.getcommissioninfo(self.data).get_leverage()
        self.margin = abs((ep * es) / leverage)


        if order.status == order.Accepted:
            print('-' * 32, ' NOTIFY ORDER ', '-' * 32)
            print('{} Order Accepted'.format(order.info['name']))
            print('{}, Status {}: Ref: {}, Size: {}, Price: {}'.format(
                date,
                order.status,
                order.ref,
                round(order.size, 2),
                'NA' if not order.price else round(order.price, 5)

            ))
            print('-' * 80)

        if order.status == order.Submitted:
            print('-' * 32, ' NOTIFY ORDER ', '-' * 32)
            print('{} Order Submitted'.format(order.info['name']))
            print('{}, Status {}: Ref: {}, Size: {}, Price: {}'.format(
                date,
                order.status,
                order.ref,
                round(order.size, 2),
                'NA' if not order.price else round(order.price, 5)

            ))
            print('-' * 80)

        if order.status == order.Completed:
            print('-' * 32, ' NOTIFY ORDER ', '-' * 32)
            print('{} Order Completed,'.format(order.info['name']))
            print('{}, Status {}: Ref: {}, Size: {}, Price: {}, Comm: {}, Margin: {}'.format(
                date,
                order.status,
                order.ref,
                round(order.size, 2),
                round(order.executed.price, 5),
                round(order.executed.comm, 2),
                round(self.margin,2)
            ))
            print('-' * 80)


        #EXPRESS EXEC---------
            if self.p.express_exec == True:
                if not self.position:
                    return

                elif order.info['name'] == 'MARKET_BUY':
                    sell_order = self.sell(size = 1000, exectype = bt.Order.Stop, price = (order.executed.price - self.atr[0]*1.5))
                    sell_order.addinfo(name = 'MARKET_BUY_STOP')
                    tp_order = self.sell(size = 1000, exectype = bt.Order.Limit, price = (order.executed.price + 0.0100), oco =sell_order)
                    tp_order.addinfo(name = 'TP_SELL')

                elif order.info['name'] == 'MARKET_SELL' :
                    sell_order = self.buy(size = 1000, exectype = bt.Order.Stop, price = (order.executed.price + self.atr[0]*1.5))
                    sell_order.addinfo(name = 'MARKET_SELL_STOP')
                    tp_order =self.buy(size = 1000, exectype = bt.Order.Limit, price = (order.executed.price - 0.0100), oco = sell_order)
                    tp_order.addinfo(name = 'TP_BUY')
        #EXPRESS EXEC---------

        if order.status == order.Canceled:
            print('-' * 32, ' NOTIFY ORDER ', '-' * 32)
            print('{} Order Cancelled'.format(order.info['name']))
            print('{}, Status {}: Ref: {}, Size: {}, Price: {}'.format(
                date,
                order.status,
                order.ref,
                round(order.size,5),
                'NA' if not order.price else round(order.price, 5)
            ))
            print('-' * 80)

        if order.status == order.Rejected:
            print('-' * 32, ' NOTIFY ORDER ', '-' * 32)
            print('{} Order Rejected'.format(order.info['name']))
            print('{}, Status {}: Ref: {}, Size: {}, Price: {}'.format(
                date,
                order.status,
                order.ref,
                round(order.size, 2),
                'NA' if not order.price else round(order.price, 5)
            ))
            print('-' * 80)

        if order.info['name'] == 'MCO':
            print('-' * 32, ' NOTIFY ORDER ', '-' * 32)
            print('WARNING: MARGIN CLOSE OUT')
            print('{}, Status {}: Ref: {}, Size: {}, Price: {}'.format(
                date,
                order.status,
                order.ref,
                round(order.size, 2),
                'NA' if not order.price else round(order.price, 5)
            ))
            print('-' * 80)
 
    # LOG TRADES------------------
    def notify_trade(self, trade):
        date = self.data.datetime.datetime()
        if trade.isclosed:
            print('-' * 32, ' NOTIFY TRADE ', '-' * 32)
            print('{}, {} Trade Price: {}, Profit, Gross {}, Net {}'.format(
                date,
                trade.data._name,
                round(trade.price, 5),
                round(trade.pnl, 2),
                round(trade.pnlcomm, 2)))
            print('-' * 80)


    def check_mco(self, value, margin_used):

        if value < (margin_used/2):
            return True
        else:
            return False

    # TRADING LOGIC-----------------
    def next(self):

        # Insert the JPY accomm
        if single_data:

            multiplier = 0.0001

            # Stoploss Calc and Units to Account values
            acct_value = self.broker.getvalue()
            risk = acct_value * self.p.pct_risk
            stop_order = round(self.atr * 1.5, 5)
            stop_order_pips = stop_order / multiplier
            pip_value = risk / stop_order_pips
            self.units = pip_value / multiplier

            if self.p.counter_currency == True:
                pip_value = risk / stop_order_pips
                self.units = pip_value / multiplier

            elif self.p.base_currency == True:
                pip_value = (risk / stop_order_pips) * self.p.ex_rate  # base currency to counter currency
                self.units = pip_value / multiplier

            elif self.p.no_currency == True:
                pip_value = (risk / stop_order_pips) / self.p.ex_rate  # counter currency to denominated account currency
                self.units = pip_value / multiplier

            if self.order:  # If any pending orders, do nothing
                return

            # Check if we are in position

            # Stop loss CONFIG
            long_sl = self.dataclose[0] - stop_order
            long_tp = self.dataclose[0] + 0.0100
            short_sl = self.dataclose[0] + stop_order
            short_tp = self.dataclose[0] - 0.0100

        #SINGLE DATA FEED

            if not self.position:

                 # Main LOGIC
                if self.trade_sig > 0:
                    buy_ord = self.buy(price=self.dataclose[0], exectype=bt.Order.Limit, transmit=False)
                    buy_ord.addinfo(name='BUY')

                    buy_ord_sl = self.sell(price=long_sl, size=buy_ord.size,
                                           exectype=bt.Order.Stop, transmit=False, parent=buy_ord)
                    buy_ord_sl.addinfo(name='BUY SL')

                    buy_ord_tp = self.sell(price=long_tp, size=buy_ord.size,
                                           exectype=bt.Order.Limit, transmit=True, parent=buy_ord)
                    buy_ord_tp.addinfo(name='BUY TP')

                    # print('RISK: {}, ATRx1.5: {}, SL: {}, TP: {}, ACCT: {}'.format(round(risk,2),
                    #                                                                round(stop_order, 5), round(long_sl, 5), round(long_tp, 5), round(acct_value, 2)))

                elif self.trade_sig < 0:
                    sell_ord = self.sell(price=self.dataclose[0],
                                         exectype=bt.Order.Limit, transmit=False)
                    sell_ord.addinfo(name='SELL')

                    sell_ord_sl = self.buy(price=short_sl, size=sell_ord.size,
                                           exectype=bt.Order.Stop, transmit=False, parent=sell_ord)
                    sell_ord_sl.addinfo(name='SELL SL')

                    sell_ord_tp = self.buy(price=short_tp, size=sell_ord.size,
                                           exectype=bt.Order.Limit, transmit=True, parent=sell_ord)
                    sell_ord_tp.addinfo(name='SELL TP')

                    # print('RISK: {}, ATRx1.5: {}, SL: {}, TP: {}, ACCT: {}'.format(round(risk,2), 
                    #     round(stop_order, 5), round(short_sl, 5), round(short_tp, 5), round(acct_value,2)))

            else:
                mco_result = self.check_mco(value = self.broker.getvalue(), margin_used = self.margin)
                if mco_result == True:
                    close = self.close()
                    close.addinfo(name = 'MCO')

                if self.p.EOE: #Exits on the end of close
                    i = list(range(0, len(self.datas)))
                    for (d,j) in zip(self.datas, i):
                        if len(d) == (d.buflen()-1):
                            exit_trade = self.close(d, exectype = bt.Order.Market)
                            exit_trade.addinfo(name = 'Exit Trade')

        #MULTIDATA FEED
        if multi_data:
            
            for i, d in enumerate(self.datas):

                #BASE SIZE VAL
                multiplier = 0.0001
                acct_value = self.broker.getvalue()
                risk = acct_value * self.p.pct_risk
                stop_order = round(self.inds[d]['atr'][0] * 1.5, 5)
                stop_order_pips = stop_order / multiplier
                pip_value = risk / stop_order_pips
                self.units = pip_value / multiplier

                #FINAL SIZE VAL (Comm Adj)
                for init, name, s in datalist:
                    if d._name[:3] == 'CAD':

                        conv_risk = risk
                        comm_spread = xom[d._name]
                        comm_new = abs((comm_spread * ((self.units/d.close[0]) * multiplier) / 2))
                        comm_adj_risk = conv_risk - (comm_new * 2)
                        pip_value_adj = comm_adj_risk / stop_order_pips
                        units_adj = pip_value_adj / multiplier

                    if d._name[3:] == 'CAD':

                        comm_spread = xom[d._name]
                        comm_new = abs((comm_spread * (self.units * multiplier) / 2))
                        comm_adj_risk = risk - (comm_new * 2)
                        pip_value_adj = comm_adj_risk / stop_order_pips
                        units_adj = pip_value_adj / multiplier

                    elif d._name[3:] != 'CAD' and d._name[:3] != 'CAD':

                        ex_rate = ex[d._name[3:]]
                        conv_risk = risk / ex_rate
                        comm_spread = xom[d._name]
                        comm_new = abs((comm_spread * ((self.units/d.close[0]) * multiplier) / 2))
                        comm_adj_risk = conv_risk - (comm_new * 2)
                        pip_value_adj = comm_adj_risk / stop_order_pips
                        units_adj = pip_value_adj / multiplier


                dt, dn = self.data.datetime.datetime(), d._name
                pos = self.getposition(d).size
                if not pos:

                    if self.inds[d]['cross'][0] == 1:

                        buy_ord = self.buy(data = d,price=d.close[0], exectype=bt.Order.Limit,size = units_adj, transmit=False)
                        buy_ord.addinfo(name='BUY')

                        buy_ord_sl = self.sell(data = d, price=d.close[0]-stop_order, size=buy_ord.size,
                                               exectype=bt.Order.Stop, transmit=False, parent=buy_ord)
                        buy_ord_sl.addinfo(name='BUY SL')

                        buy_ord_tp = self.sell(data = d, price=d.close[0]+stop_order, size=buy_ord.size,
                                               exectype=bt.Order.Limit, transmit=True, parent=buy_ord)
                        buy_ord_tp.addinfo(name='BUY TP')

                        comminfo = forexSpreadCommissionScheme(spread=comm_spread, data = d._name, interest=0.03, ex_rate = ex_rate)  
                        self.broker.addcommissioninfo(comminfo)

                        # print('RISK: {}, ATRx1.5: {}, ACCT: {}, {}, {}, {} {} {} sl:{}, tp:{}'.format(round(risk,2),round(stop_order, 5), 
                        #     round(acct_value, 2), ex_rate,d._name,comm_spread, d.close[0], conv_risk, d.close[0]-stop_order, d.close[0]+stop_order))

                    elif self.inds[d]['cross'][0] == -1:

                        sell_ord = self.sell(data = d, price=d.close[0],
                                             exectype=bt.Order.Limit, size = units_adj, transmit=False)
                        sell_ord.addinfo(name = 'SELL')

                        sell_ord_sl = self.buy(data = d, price=d.close[0]+stop_order, size=sell_ord.size,
                                               exectype=bt.Order.Stop, transmit=False, parent=sell_ord)
                        sell_ord_sl.addinfo(name = 'SELL SL')

                        sell_ord_tp = self.buy(data = d, price=d.close[0]-stop_order, size=sell_ord.size,
                                               exectype=bt.Order.Limit, transmit=True, parent=sell_ord)
                        sell_ord_tp.addinfo(name = 'SELL TP')
                        comminfo = forexSpreadCommissionScheme(spread=comm_spread, data = d._name, interest=0.03, ex_rate = ex_rate)  
                        self.broker.addcommissioninfo(comminfo)

                        # print('RISK: {}, ATRx1.5: {}, ACCT: {}, {}, {}, {} {} {}, sl:{}, tp:{}'.format(round(risk,2),round(stop_order, 5), 
                        #     round(acct_value, 2), ex_rate,d._name,comm_spread, d.close[0], conv_risk, d.close[0]+stop_order, d.close[0]-stop_order))

                #MARGIN CALL            
                else:

                    mco_result = self.check_mco(value = self.broker.getvalue(), margin_used = self.margin)
                    if mco_result == True:
                        close = self.close()
                        close.addinfo(name = 'MCO')

        #MARKET EXEC (ORDER DATA)
        if self.p.express_exec == True:

            if not self.position:

                if self.trade_sig < 0:
                        short_order = self.sell(size = 1000)
                        short_order.addinfo(name = 'MARKET_SELL')

                elif self.trade_sig > 0:
                        buy_order = self.buy(size = 1000)
                        buy_order.addinfo(name='MARKET_BUY')

            else:
                #MARGIN CALL
                mco_result = self.check_mco(value = self.broker.getvalue(), margin_used = self.margin)
                if mco_result == True:
                    close = self.close(size = 1000)
                    close.addinfo(name = 'MCO')

                if self.p.EOE: # Exits on end with Market
                    i = list(range(0, len(self.datas)))
                    for (d,j) in zip(self.datas, i):
                        if len(d) == (d.buflen()-1):
                            exit_trade = self.close(d, exectype = bt.Order.Market)
                            exit_trade.addinfo(name = 'Exit Trade')

# COMMISSION SCHEME----------------
class forexSpreadCommissionScheme(bt.CommInfoBase):
    params = (

        ('spread', 5.0),
        ('ex_rate', 0.70),
        ('stocklike', False),
        ('JPY_pair', False),
        ('counter_currency', False),
        ('commtype', bt.CommInfoBase.COMM_FIXED),
        ('leverage', 50.0),
        ('automargin', False),
        ('base_currency', False),
        ('no_currency', True),
        ('interest_long', True),
        ('interest', 0.03),
        ('data','data_name')

    )

    def _getcommission(self, size, price, pseudoexec):
        if multi_data:
            if self.params.JPY_pair == True:
                multiplier = 0.01
            else:
                multiplier = 0.0001

            if self.params.data[3:] == 'CAD':
                comm = abs((self.params.spread * (size * multiplier) / 2))

            elif self.params.data[:3] == 'CAD':
                comm = abs((self.params.spread * ((size / price) * multiplier) / 2))

            elif self.params.data[3:] != 'CAD' and self.params.data[:3] != 'CAD':
                comm = abs((self.params.spread * ((size / price) * multiplier) / 2))*self.p.ex_rate

            return comm

        else:

            if self.params.JPY_pair == True:
                multiplier = 0.01
            else:
                multiplier = 0.0001

            if self.params.counter_currency == True:
                comm = abs((self.params.spread * (size * multiplier) / 2))

            elif self.params.base_currency == True:
                comm = abs((self.params.spread * ((size / price) * multiplier) / 2))

            elif self.params.no_currency == True:
                comm = abs((self.params.spread * ((size / price) * multiplier) / 2))*self.p.ex_rate

            return comm 

    def get_leverage(self):

        return self.p.leverage

    def getsize(self, price, cash):

        if not self._stocklike:

            return int(self.p.leverage * (cash // self.get_margin(price)))
        return int(self.p.leverage * (cash // price))

    def _get_credit_interest(self, data, size, price, days, dt0, dt1):
        return 1.0 * abs(size) * price * (self.p.interest / 365.0)


class RiskSizer(bt.Sizer):

    """ Sizing based on stop loss and ATR
    formula:

    risk = cash * 0.02
    atrX1.5 pips
    pip_dollar = risk/ atrX1.5
    units = pip_dollar/0.0001

    COMM_ADJ UNITS

    comm_adj_risk = risk - (comm_scheme * 2)
    pip_dollar = comm_adj_risk / atrX1.5
    comm_units = pip_dollar / 0.0001

     """

    params = (('risk',0.02),
            ('counter_currency', False),
            ('no_currency',True),
            ('base_currency',False),
            ('ex_rate', 5.0),
            ('spread',5.0),)

    def _getsizing(self, comminfo, cash, data, isbuy):

        multiplier = 0.0001
        atr = self.strategy.atr
        size = 0
        acct_value = cash
        max_risk = math.floor(acct_value * self.p.risk)
        stop_order = (atr*1.5)
        stop_order_pips = stop_order / multiplier
        pip_value = max_risk / stop_order_pips
        units = pip_value / multiplier

        if self.params.counter_currency == True:
                comm_new = abs((self.params.spread * (units * multiplier) / 2))

        elif self.params.base_currency == True:
                comm_new = abs((self.params.spread * ((units / data.close) * multiplier) / 2))

        elif self.params.no_currency == True:
                comm_new = abs((self.params.spread * ((units / data.close) * multiplier) / 2))*self.p.ex_rate

        comm_adj_risk = max_risk - (comm_new * 2)
        pip_value_adj = comm_adj_risk / stop_order_pips
        units_adj = pip_value_adj / multiplier

        if comm_adj_risk < 0:
            return 0 

        if isbuy == True:
            comm_adj_size = units_adj
        else:
            comm_adj_size = units_adj * -1

        comm_adj_size = math.floor(comm_adj_size)

        return comm_adj_size

# ANALYZERS-----------
class trade_list(bt.Analyzer):

    def get_analysis(self):

        return self.trades

    def __init__(self):
        self.trades = []
        self.cumprofit = 0.0

    def notify_trade(self, trade):

        if trade.isclosed:

            acct_value = self.strategy.broker.getvalue()

            dir = 'short'

            if trade.history[0].event.size > 0:
                dir = 'long'

            pricein = trade.history[len(trade.history) - 1].status.price
            priceout = trade.history[len(trade.history) - 1].event.price
            datein = bt.num2date(trade.history[0].status.dt)
            dateout = bt.num2date(trade.history[len(trade.history) - 1].status.dt)

            if trade.data._timeframe >= bt.TimeFrame.Days:
                datein = datein.date()
                dateout = dateout.date()

            pctchange = 100 * priceout / pricein - 100

            pnl = trade.history[len(trade.history) - 1].status.pnlcomm

            pnlpcnt = 100 * pnl / acct_value

            barlen = trade.history[len(trade.history) - 1].status.barlen

            pbar = pnl / barlen

            trade_name = trade.data._name

            self.cumprofit += pnl

            size = value = 0.0
            for record in trade.history:
                if abs(size) < abs(record.status.size):
                    size = record.status.size
                    value = record.status.value
                highest_in_trade = max(trade.data.high.get(ago=0, size=barlen + 1))
                lowest_in_trade = min(trade.data.low.get(ago=0, size=barlen + 1))
                hp = 100 * (highest_in_trade - pricein) / pricein
                lp = 100 * (lowest_in_trade - pricein) / pricein

                if dir == 'long':
                    mfe = hp
                    mae = lp
                if dir == 'short':
                    mfe = -lp
                    mae = -hp

                self.trades.append({'ref': trade.ref, 'dir': dir,
                                    'datein': datein, 'pricein': pricein, 'dateout': dateout,
                                    'priceout': priceout, 'pctchnge%': round(pctchange, 2), 'pnl': pnl,
                                    'pnl%': round(pnlpcnt, 2), 'size': size, 'value': value, 'cumpnl': self.cumprofit,
                                    'nbars': barlen, 'pnl/bar': round(pbar, 2), 'mfe%': round(mfe, 2), 'mae%': round(mae, 2), 'Data':trade_name})

class TimeSeriesSplitImproved(TimeSeriesSplit):
    """Time Series cross-validator
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide `.
    Parameters
    ----------
    n_splits : int, default=3
        Number of splits. Must be at least 1.
    Examples
    --------
    >>> from sklearn.model_selection import TimeSeriesSplit
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([1, 2, 3, 4])
    >>> tscv = TimeSeriesSplit(n_splits=3)
    >>> print(tscv)  # doctest: +NORMALIZE_WHITESPACE
    TimeSeriesSplit(n_splits=3)
    >>> for train_index, test_index in tscv.split(X):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0] TEST: [1]
    TRAIN: [0 1] TEST: [2]
    TRAIN: [0 1 2] TEST: [3]
    >>> for train_index, test_index in tscv.split(X, fixed_length=True):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0] TEST: [1]
    TRAIN: [1] TEST: [2]
    TRAIN: [2] TEST: [3]
    >>> for train_index, test_index in tscv.split(X, fixed_length=True,
    ...     train_splits=2):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0 1] TEST: [2]
    TRAIN: [1 2] TEST: [3]
 
    Notes
    -----
    When ``fixed_length`` is ``False``, the training set has size
    ``i * train_splits * n_samples // (n_splits + 1) + n_samples %
    (n_splits + 1)`` in the ``i``th split, with a test set of size
    ``n_samples//(n_splits + 1) * test_splits``, where ``n_samples``
    is the number of samples. If fixed_length is True, replace ``i``
    in the above formulation with 1, and ignore ``n_samples %
    (n_splits + 1)`` except for the first training set. The number
    of test sets is ``n_splits + 2 - train_splits - test_splits``.
    """
 
    def split(self, X, y=None, groups=None, fixed_length=False,
              train_splits=1, test_splits=1):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        # X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
        # y = np.array([1, 2, 3, 4, 5, 6])
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like, with shape (n_samples,), optional
            Always ignored, exists for compatibility.
        fixed_length : bool, hether training sets should always have
            common length
        train_splits : positive int, for the minimum number of
            splits to include in training sets
        test_splits : positive int, for the number of splits to
            include in the test set
        Returns
        -------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        train_splits, test_splits = int(train_splits), int(test_splits)
        if n_folds > n_samples:
            raise ValueError(
                ("Cannot have number of folds ={0} greater"
                 " than the number of samples: {1}.").format(n_folds,
                                                             n_samples))
        if (n_folds - train_splits - test_splits) < 0 and test_splits > 0:
            raise ValueError(
                ("Both train_splits and test_splits must be positive"
                 " integers."))
        indices = np.arange(n_samples)
        split_size = (n_samples // n_folds)
        test_size = split_size * test_splits
        train_size = split_size * train_splits
        test_starts = range(train_size + n_samples % n_folds,
                            n_samples - (test_size - split_size),
                            split_size)
        if fixed_length:
            for i, test_start in zip(range(len(test_starts)),
                                     test_starts):
                rem = 0
                if i == 0:
                    rem = n_samples % n_folds
                yield (indices[(test_start - train_size - rem):test_start],
                       indices[test_start:test_start + test_size])
        else:
            for test_start in test_starts:
                yield (indices[:test_start],
                    indices[test_start:test_start + test_size])


# SQN---------------
def printTradeAnalysis(analyzer):

    total_open = analyzer.total.open

    total_closed = analyzer.total.closed

    total_won = analyzer.won.total

    total_lost = analyzer.lost.total

    win_streak = analyzer.streak.won.longest

    lose_streak = analyzer.streak.lost.longest

    long_total = analyzer.long.total

    short_total = analyzer.short.total

    pnl_net = round(analyzer.pnl.net.total, 2)

    strike_rate = round((total_won / total_closed) * 100, 2)
    h1 = ['Total Open', 'Total Closed', 'Total Won', 'Total Lost', 'Long Total']
    h2 = ['Strike Rate', 'Win Streak', 'Losing Streak', 'PnL Net', 'Short Total']
    r1 = [total_open, total_closed, total_won, total_lost, long_total]
    r2 = [strike_rate, win_streak, lose_streak, pnl_net, short_total]

    if len(h1) > len(h2):
        header_length = len(h1)
    else:
        header_length = len(h2)
    print_list = [h1, r1, h2, r2]
    row_format = "{:<15}" * (header_length + 1)
    print("Trade Analysis Results:")
    for row in print_list:
        print(row_format.format('', *row))


def printSQN(analyzer):
    sqn = round(analyzer.sqn, 2)
    print('SQN: {}'.format(sqn))


class MonteCarlo(bt.Analyzer):


  params = (('start', 100000),
              ('end',375000),
              ('margin', 50000),
              ('iter',2500),
              ('lotsize',1))

  def __init__(self):
    self.trade_list = []
    self.step = self.p.start/4
    self.rand_val = 0.0
    self.data = pd.DataFrame()
    

  def notify_trade(self, trade):

    if trade.isclosed:
      self.trades = trade.pnlcomm
      self.trade_list.append(self.trades)
      
  def get_analysis(self):
    df = pd.DataFrame(self.trade_list)
    lotsize = 1
    
    for i in range(self.p.start, self.p.end, int(self.step)):
      
      max_equity = i
      drawdown = 0
      ruin_list = []
      val_list = []
      drawdown_list = []
      return_list = []
      dd_ret_list = []
      equity_list = []
      diff_list = []

      for trades in range(2501):
        random.shuffle(self.trade_list)
        trades  = random.choice(self.trade_list)
        equity = trades + i
        equity_list.append(equity)

      for val in equity_list:
          ret_ = ((val/i)-1)*100
          return_list.append(ret_)
          diff = val - i
          diff_list.append(diff)

          if val > max_equity:
            max_equity = val
          else:
            dd = 1-(val/max_equity)
            if dd > drawdown:
              drawdown = dd
              drawdown_list.append(drawdown)
          if drawdown != 0:
              dd_ret = ((val / i)-1)/drawdown
              dd_ret_list.append(dd_ret)
          if val < self.p.margin:
            ruin_list.append(val)

      ruin = int(len(ruin_list)/self.p.iter)
      if ruin == 1:
        print('Ruined')
        continue
      drawdown = np.median(drawdown_list)
      return_ = np.median(return_list)
      equity_median = np.median(equity_list)
      drawdown_ret_ratio = np.median(dd_ret_list)
      var = np.var(equity_list)
      stdev = math.sqrt(var)
      median_profit = equity_median - i
      win = []
      lose = []
      for equity in equity_list:
        if equity > i:
          win.append(equity)
        elif equity < i:
          lose.append(equity)

      win_pct = ((len(win) / len(equity_list)))*100
      loss_pct = ((len(lose) / len(equity_list)))*100

      df3 = pd.DataFrame({
                        'Initial Equity': i,
                        'Final Equity': equity_median,
                        'Median Profit': median_profit,
                        'Return':return_,
                        'Win%':win_pct,
                        'Loss%':loss_pct,
                        'Drawdown': (drawdown)*100,
                        'Return/DD': drawdown_ret_ratio,
                        'Standard Deviation': stdev,
                        'Variance': var}, index = [0], columns = ['Initial Equity',
                        'Final Equity','Median Profit','Return','Win%','Loss%','Drawdown','Return/DD','Standard Deviation','Variance'])
      self.data = self.data.append(df3)

    data_csv = self.data.to_csv('MonteCarlo.csv')
    print('Data Recorded')


def df_converter(csv):
    df = pd.read_csv(csv)
    df['date'] = pd.to_datetime([x[:-21] for x in df['date'].squeeze().tolist()], dayfirst=True)
    df.drop_duplicates(subset='volume', keep=False, inplace=True)
    df.set_index('date', inplace=True)
    return df

class AcctStats(bt.Analyzer):
 
    def __init__(self):
        self.start_val = self.strategy.broker.get_value()
        self.end_val = None
 
    def stop(self):
        self.end_val = self.strategy.broker.get_value()
 
    def get_analysis(self):
        return {"start": self.start_val, "end": self.end_val,
                "growth": self.end_val - self.start_val, "return": self.end_val / self.start_val}

if init_wf == True:

    tscv = TimeSeriesSplitImproved(n_splits=10)
    split = tscv.split(df.index, fixed_length = True, train_splits = 5, test_splits = 1)
    cerebro = bt.Cerebro()
    init_port = cerebro.broker.getvalue()

    walk_forward_results = list()
    in_sample = list()
    
    for train, test in split:

        windowset = set()
        while len(windowset) < 40:
            f = random.randint(1, 10) * 5
            s = random.randint(1, 10) * 10
            if f > s:    # Cannot have the fast moving average have a longer window than the slow, so swap
                f, s = s, f
            elif f == s:    # Cannot be equal, so do nothing, discarding results
                continue
            windowset.add((f, s))

        windows = list(windowset)

        trainer = bt.Cerebro(stdstats=False, maxcpus=1)
        trainer.broker.set_cash(1000000)
        trainer.addanalyzer(AcctStats)
        trainer.addanalyzer(bt.analyzers.TradeAnalyzer, _name = 'ta')
        tester = deepcopy(trainer)

        #TRAINING
        trainer.optstrategy(TestStrategy, optim=True, optim_fs=windows)
        data  = bt.feeds.PandasData(dataname=df.iloc[train], name = 'Train')
        trainer.adddata(data)
        res = trainer.run()
        res_dict = dict()
        res_dict["start_date_optimized"] = df.iloc[train[0]].name
        res_dict["end_date_optimized"] = df.iloc[train[-1]].name

        opt_res = pd.DataFrame({r[0].params.optim_fs: r[0].analyzers.acctstats.get_analysis() for r in res}
                           ).T.loc[:, "return"].sort_values(ascending=False).index[0]
        res_dict["fast"], res_dict["slow"] = opt_res
        for r in res:
            res_dict = r[0].analyzers.acctstats.get_analysis()
        in_sample.append(res_dict)

        

        #TESTING
        tester.addstrategy(TestStrategy, optim = True, optim_fs = opt_res)
        data  = bt.feeds.PandasData(dataname=df.iloc[test], name = 'Test')
        tester.adddata(data)

        res = tester.run()
        res_dict = res[0].analyzers.acctstats.get_analysis()
        res_dict["fast"], res_dict["slow"] = opt_res

        res_dict["start_date"] = df.iloc[test[0]].name
        res_dict["end_date"] = df.iloc[test[-1]].name
        walk_forward_results.append(res_dict)


        # tester.addstrategy(TestStrategy)
        # data = bt.feeds.PandasData(dataname=df.iloc[test], name = 'Test')
        # tester.adddata(data)
        # res = tester.run()
        # res_dict = res[0].analyzers.acctstats.get_analysis()
        # res_dict["start_date_Walk_Forward"] = df.iloc[test[0]].name
        # res_dict["end_date_Walk_Forward"] = df.iloc[test[-1]].name
        # walk_forward_results.append(res_dict)

    print('-'*10)
    wfdf = pd.DataFrame(walk_forward_results)
    ifdw = pd.DataFrame(in_sample)
    frames = [wfdf,ifdw]
    results = pd.concat(frames, axis = 1)
    print(results)
    export = results.to_csv('WalkAnalysis.csv')

#NORMAL TEST
if __name__ == '__main__':
    cerebro = bt.Cerebro()
    
    cerebro.addstrategy(TestStrategy)  # add strategy to Cerebro

    #cerebro.broker = bt.brokers.BackBroker(slip_fixed=1)  
    cerebro.broker = bt.brokers.BackBroker(slip_perc=0.001,slip_limit = False) # Account for slippage 0.1%

    cash_val = cerebro.broker.setcash(100000)  # Initial Cash

    # Initial Portfolio Value
    initial_port = print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())  # retrieve Portfolio Inital Value

    #DATAFEED RAW
    datalist = [(r'C:\Users\Robin\Dropbox\CSV Files\EURUSD_HR.csv','AUDUSD',0.7), ('/Users/robindhillon/Desktop/EURCAD.csv','EURCAD',3.6),
                    ('/Users/robindhillon/Desktop/EURUSD.csv','EURUSD',1.2), ('/Users/robindhillon/Desktop/USDCHF.csv','USDCHF',1.7),
                    ('/Users/robindhillon/Desktop/AUDCAD.csv','AUDCAD',1.7), ('/Users/robindhillon/Desktop/EURGBP.csv','EURGBP',1.4)]

    ex = {'USD':1.34102,'AUD': 0.95043, 'EUR':1.50825, 'CHF':1.3479,'GBP':1.76342}
    xom = {'AUDUSD':0.7, 'EURCAD':3.6,'EURUSD':1.2,'USDCHF':1.7,'AUDCAD':1.7,'EURGBP':1.4}

    if multi_data:

        for i, d, spread in datalist:
            df = df_converter(i)
            data = bt.feeds.PandasData(dataname=df)
            cerebro.adddata(data, name = d)
        
        cerebro.addanalyzer(trade_list, _name='trade_list')
        cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='ta')

        #MONTE CARLO SIM
        if monte_test:
            cerebro.addanalyzer(MonteCarlo, _name = 'montecarlo')
        
        # Cerebro RUN
        strats = cerebro.run(tradehistory=True)

        # Analyzer Print
        trade_list = strats[0].analyzers.trade_list.get_analysis()
        print(tabulate(trade_list, headers='keys'))

        printTradeAnalysis(strats[0].analyzers.ta.get_analysis())
        printSQN(strats[0].analyzers.sqn.get_analysis())

        # Final Portfolio Value
        print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
        # Plot Function
        #cerebro.plot(style='candlestick')

    else:

        comminfo = forexSpreadCommissionScheme(spread=0.7, no_currency=True, interest=0.03, ex_rate = 1.34102)  # Commision using spreads and Credit Rate
        cerebro.addsizer(RiskSizer,spread = 0.7, no_currency = True, ex_rate = 1.34102)
        cerebro.broker.addcommissioninfo(comminfo)

        data = bt.feeds.PandasData(dataname=df_converter(datalist[0][0]), fromdate=datetime(2015, 1, 1), todate=datetime(2017, 5, 1))  # Datafeed with date limit
        cerebro.adddata(data)

        # Analyzer Call
        cerebro.addanalyzer(trade_list, _name='trade_list')
        cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='ta')
        

        #MONTE CARLO SIM
        if monte_test:
            cerebro.addanalyzer(MonteCarlo, _name = 'montecarlo')
        
        # Cerebro RUN
        strats = cerebro.run(runonce=True,tradehistory=True)

        # Analyzer Print
        trade_list = strats[0].analyzers.trade_list.get_analysis()
        print(tabulate(trade_list, headers='keys'))

        printTradeAnalysis(strats[0].analyzers.ta.get_analysis())
        printSQN(strats[0].analyzers.sqn.get_analysis())

        # Final Portfolio Value
        print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
        # Plot Function
        #cerebro.plot(style='candlestick')



