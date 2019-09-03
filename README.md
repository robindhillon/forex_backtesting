# forex_backtesting

## Getting Started

To start you will obviously need a data file handy. Maninly it should be in the CSV format and ideally consist of OHLCV data(open,high,low,close,volume). A good resource for data is dukascopy.com, though the data sometimes can be a little messy, it still is a good free data source. It is also recommended to run this application on a decent system, to save some run time and optimize quicker.

### Prerequisites

You would need the following packages and versions:
1. Python 3.5 or lower (you can download anaconda and setup a custom environment and install the dependencies)
2. Backtrader
```
pip install backtrader
```
3. Pandas
```
pip install pandas
```
4. Numpy
```
pip install numpy
```
5. Sklearn
```
pip install sklearn
``` 
6. tabulate
```
pip install tabulate
```

### How to get the program running

As mentinoned before you want to have a good datastream or a csv file ready. The strategey has several parameters which can be tuned to simulate different market scenarios.

1. Set your starting Cash amount
```
cash_val = cerebro.broker.setcash(100000)  # Initial Cash
```

2. Load the data: first thing first, get the data file location and inoput it in the datalist at the bottom of the program
```
r'C:\Users\Robin\Dropbox\CSV Files\EURUSD_HR.csv
```
*if you are using multiple data files, input all the files in the datalist

3. Add the data with the adddata call
*if using a single data file the file can be intialized as:
```
dataname=df_converter(datalist[0][0])
```
4. Rest is just setting parameters like commissions and slippage to your preference
*the commission scheme is based on the OANDA commission scheme
https://www1.oanda.com/register/docs/divisions/oc/price_sheet.pdf

You should end up with data like this after you run the program
```
 pctchnge%    pnl/bar    ref    mae%  Data      priceout    value    mfe%        pnl  datein                 size  dir      pnl%      cumpnl  dateout                nbars    pricein
-----------  ---------  -----  ------  ------  ----------  -------  ------  ---------  -------------------  ------  -----  ------  ----------  -------------------  -------  ---------
      -0.89      -1.68      1   -0.89             1.17273    10000    0.33  -109.39    2015-01-09 11:00:00   10000  long    -1.11  -109.39     2015-01-14 04:00:00       65    1.18321
      
      -0.95     -14.17      2   -1.77             1.16642    10000    0.1   -113.344   2015-01-14 20:00:00   10000  long    -1.16  -222.734    2015-01-15 04:00:00        8    1.17759

```


## Running the tests

I have included some features that can help you conduct tests for further analysis besides the already included trades. At the top of the program, you can intialize features such as Walk-forward analysis, Monte Carlo Simulations and multiple data feeds. Other tweakable features include:
1. Commission spread
2. Leverage
3. overnight holding interest
4. order types
5. Order Notification 

## Built With
1. www.backtest-rookies.com
2. www.backtrader.com
 

## Authors

* **Robin Dhillon** - *Initial work* - [robinbeatrix](https://github.com/robinbeatrix)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to the users on backtrader's community page (@ab_trader, @backtrader)

