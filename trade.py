from ast import Add
from turtle import pos, position
from sqlalchemy import null
from urllib3 import response
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tda import auth, client
from tda.orders.common import Duration, Session
from tda.orders.equities import equity_buy_limit
from tda.orders.options import OptionSymbol
import json
import config
import datetime
import time
import trade
import pytz
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc
from scipy import signal

import gtts
from playsound import playsound

# This module is imported so that we can play the converted audio
import os
  
# The text that you want to convert to audio
mytext = 'Running Algo!'
  
# Language in which you want to convert
language = 'en'
  
# Passing the text and language to the engine, here we have marked slow=False. Which tells 
# the module that the converted audio should have a high speed
myobj = gtts.gTTS(text=mytext, lang=language, slow=False)
  
# Saving the converted audio in a mp3 file named welcome 
myobj.save("welcome.mp3")
  
# Playing the converted file
playsound("welcome.mp3")


# Chart Studio Plotly
import chart_studio
plotly_cs_username = 'jonflees'
plotly_cs_api_key = 'jSdnOLzdG468sp29zVuA'

chart_studio.tools.set_credentials_file(username=plotly_cs_username, api_key=plotly_cs_api_key)
import chart_studio.plotly as py
import chart_studio.tools as tls

# Connecting to Database
import mysql.connector

db = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="F!335m0de",
        database="tda_master_database")

mycursor = db.cursor()

print("\n********** Connected to {} **********".format(db.database))


try:
        c = auth.client_from_token_file(config.token_path, config.api_key)
except FileNotFoundError:
        from selenium import webdriver
        with webdriver.Chrome(executable_path='/Users/jonflees/FLEES-ANALYTICS-LOCAL/tdameritrade/chromedriver') as driver:
                c = auth.client_from_login_flow(
                        driver, config.api_key, config.redirect_uri, config.token_path)


def get_response(symbol, today, next_friday):

        response = c.get_option_chain(symbol, contract_type=c.Options.ContractType.PUT, strike_count=10,
                                        from_date=today, to_date=next_friday) #strike_range=c.Options.StrikeRange.OUT_OF_THE_MONEY 

        return response


def stonk(response, SP, dte, exp_dates, portfolio_value, strike_key_errors):

        stonk_list = []

        try:

                for i in exp_dates:

                        #print(SP)      # use to find if a certain ticker isnt working

                        # Only consider if OI is at least 50
                        if response.json()["putExpDateMap"][i][SP][0]["openInterest"] > 10:

                                desc = response.json()["putExpDateMap"][i][SP][0]["description"]
                                symbol = response.json()["symbol"]
                                #print(symbol)  # use only with above print statement
                                strike = response.json()["putExpDateMap"][i][SP][0]["strikePrice"]
                                price = round(response.json()["underlyingPrice"], 3)
                                bid = response.json()["putExpDateMap"][i][SP][0]["bid"]
                                breakeven = round(strike - bid, 2)
                                contracts = round(portfolio_value / strike / 100) # cash secured calcs only
                                maxProfit = round(float(contracts * bid * 100), 2)
                                maxLoss = (strike * contracts * 100) - maxProfit
                                pctToStrike = round((strike - price) * 100/ price, 2)
                                pctToBE = round((breakeven - price) * 100/ price, 2)
                                delta = round(float(response.json()["putExpDateMap"][i][SP][0]["delta"]), 3)
                                delta = -.01 if delta == 'NaN' else delta   # convert NaN to 1%
                                tvol = round(float(response.json()["putExpDateMap"][i][SP][0]["volatility"]) * np.sqrt(dte / 253), 3)
                                profitDTE = round(maxProfit / dte, 2) if dte > 0 else 0 
                                rr_1st = round(profitDTE * pctToBE * -1, 2)  # rr_1st gets best profitDTE to pctBE ratio
                                
                                desc = desc.split()
                                desc = " ".join(desc[0:6])

                                stonk_list.append([desc,symbol,dte,strike,price,bid,breakeven,contracts,maxProfit,maxLoss,pctToStrike,
                                        pctToBE,delta,tvol,profitDTE,rr_1st])

        except KeyError:
                #print('KeyError Encounted (Strike)')
                strike_key_errors += 1

        return stonk_list, strike_key_errors


# This is what to update, use ML (back propogation?)
def get_risk_reward(maxProfit, expProfit, probOTM, probBE, dte, pctToBE):

        if dte > 0 and pctToBE < 0:

                rr = round( expProfit * np.log(-1*pctToBE / dte), 3)

        elif dte > 0 and pctToBE > 0:
                
                rr = round( expProfit * -1 * np.log(pctToBE / dte), 3)

        else:
                rr = 0.001
        
        # ((maxProfit*probOTM/100) + 
        # CURRENT IS GOOD BUT NEEDS WORK EVENTUALLY

        return rr


def MonteCarlo(df):

        NUM_SIMULATIONS =  26000

        probOTM_list = []
        probBE_list = []
        expProfit_list = []
        #expLoss_list = []
        rr2_list = []

        for y in range(len(df)):

                price = df.iloc[y]['Price']
                strike = df.iloc[y]['Strike']
                tVol = float(df.iloc[y]['TVol'])
                breakeven = df.iloc[y]['Breakeven']
                contracts = df.iloc[y]['Contracts']
                maxProfit = df.iloc[y]['MaxProfit']
                #delta = df.iloc[y]['Delta']
                dte = int(df.iloc[y]['DTE'])
                #rr1 = df.iloc[y]['RR1.0']
                pctToBE = df.iloc[y]['PctToBE']

                last_price_list = []
                countOTM = 0
                countBE = 0

                dte = dte - 2 if dte > 7 else dte   # remove weekend bc no trading

                for _ in range(NUM_SIMULATIONS):

                        # consider updating '0' mean to something tied to the state of market & beta
                        pred_last_price = round(price * (1 + np.random.normal(0, tVol/100)), 4)
              
                        last_price_list.append(pred_last_price)
                        countOTM = countOTM + 1 if pred_last_price > strike else countOTM
                        countBE = countBE + 1 if pred_last_price > breakeven else countBE

                expectedProfit = 0
                expectedLoss = maxProfit * -1

                for j in last_price_list:

                        if j < strike and j > breakeven:
                                expectedProfit += ((j - breakeven - .05)*100*contracts)  # -.05 to account for pottential buy back
                        elif j < breakeven: 
                                expectedLoss += ((j - breakeven - .05)*-100*contracts )  # -.05 to account for pottential buy back, .05 is spread (this may need dynamically updated later)
                        else:
                                expectedProfit += maxProfit - (.05*100*contracts)
                
                # probOTM is the average of prob(NumSims)--local and prob(from delta)--market
                probOTM_list.append( round( 100*countOTM/NUM_SIMULATIONS, 2) ) #+ (1+delta))/2
                probBE_list.append( round( 100*countBE/NUM_SIMULATIONS, 2) )
                expProfit_list.append( round(expectedProfit / NUM_SIMULATIONS, 2) )
                #expLoss_list.append( round(expectedLoss / NUM_SIMULATIONS, 2) )
                rr2_list.append( get_risk_reward(maxProfit, expProfit_list[y], probOTM_list[y], probBE_list[y], dte, pctToBE ) )
                
        df['ProbOTM'] = probOTM_list
        df['ProbBE'] = probBE_list
        df['ExpProfit'] = expProfit_list
        #df['ExpLoss'] = expLoss_list
        df['RR2.0'] = rr2_list

        return df



def getExpDates():

        # gets EST Time, get_exp_dates() function
        tz_NY = pytz.timezone('America/New_York') 
        today = datetime.datetime.now(tz_NY).date()
        friday = today + datetime.timedelta((4-today.weekday()) % 7)
        next_friday = today + datetime.timedelta(((4-today.weekday()) % 7)+7)
        this_num_days = friday - today
        next_num_days = next_friday - today

        this_exp_date = str(friday) + str(":") + str(this_num_days.days)    
        next_exp_date = str(next_friday) + str(":") + str(next_num_days.days)       
        exp_dates = [this_exp_date, next_exp_date]

        return exp_dates, today, next_friday



def get_all(symbol_list, portfolio_value):

        # gets EST Time, get_exp_dates() function
        tz_NY = pytz.timezone('America/New_York') 
        today = datetime.datetime.now(tz_NY).date()
        friday = today + datetime.timedelta((4-today.weekday()) % 7)
        next_friday = today + datetime.timedelta(((4-today.weekday()) % 7)+7)
        this_num_days = friday - today
        next_num_days = next_friday - today

        this_exp_date = str(friday) + str(":") + str(this_num_days.days)    
        next_exp_date = str(next_friday) + str(":") + str(next_num_days.days)       
        exp_dates = [this_exp_date, next_exp_date]

        # DataFrame for symbol_list
        column_names = ['Symbol','DTE','Strike','Price','Bid','Breakeven','Contracts','MaxProfit','MaxLoss',
                'PctToStrike','PctToBE','Delta','TVol','ProfitDTE','RR1.0','Description']

        df = pd.DataFrame(columns = column_names, dtype=object)

        strike_key_errors, exp_date_key_errors = 0,0

        for i in range(len(symbol_list)):

                sym_response = get_response(symbol_list[i], today, next_friday)
                #print(json.dumps(sym_response.json(), indent=4)) # shows full json text for each strike

                for j in range(len(exp_dates)):

                        try:

                                # strikes are k
                                for k in sym_response.json()["putExpDateMap"][exp_dates[j]]:

                                        dte = int(exp_dates[j][11:])

                                        stonk_response, strike_key_errors = stonk(sym_response, k, dte, exp_dates, portfolio_value, strike_key_errors)

                                        # Avoid KeyError and Empty List results
                                        if bool(stonk_response) and len(stonk_response) > 1:

                                                stonk_df = {'Symbol': stonk_response[j][1],'DTE': stonk_response[j][2],'Strike': stonk_response[j][3], 
                                                'Price': stonk_response[j][4], 'Bid': stonk_response[j][5], 'Breakeven': stonk_response[j][6],
                                                'Contracts': stonk_response[j][7], 'MaxProfit': stonk_response[j][8],'MaxLoss': stonk_response[j][9],
                                                'PctToStrike': stonk_response[j][10],'PctToBE': stonk_response[j][11],'Delta': stonk_response[j][12], 
                                                'TVol': stonk_response[j][13],'ProfitDTE': stonk_response[j][14],
                                                'RR1.0': stonk_response[j][15],'Description': stonk_response[j][0]}
                                                # keep description as last column always

                                                df = df.append(stonk_df, ignore_index = True)
                        except KeyError:
                                exp_date_key_errors += 1
                                #print('KeyError Encounted (Exp Date)')
        print('\nStrike KeyErrors: {}\nExp Date KeyErrors: {}'.format(strike_key_errors,exp_date_key_errors))
        return df


# risk reward = profit*coef1 + maxloss*coef2 + pctStrike*coef3 + pctBE*coef4
# regression?
# maximization of profit and minimization of loss.... where is threshold
# threshold score will equate to risk reward value

def get_account_info():
        
        # ACCOUNT INFORMATION
        acc_response = c.get_account(config.account_id)
        acc_positions_response = c.get_account(account_id=config.account_id, fields=c.Account.Fields.POSITIONS)
        acc_orders_response = c.get_account(account_id=config.account_id, fields=c.Account.Fields.ORDERS)

        #print(json.dumps(acc_positions_response.json(), indent=4))

        column_names = ['Position','OptionSymbol','Symbol','DTE','Strike','Price','AvgCost','Breakeven','Contracts','MaxProfit','Bid','Ask','Mark','PctToStrike','PctToBE','TVol']
        position_df = pd.DataFrame(columns=column_names, dtype=object)

        # Only executes if there's an active position
        try: 
                  
                for i in range(len(acc_positions_response.json()["securitiesAccount"]["positions"])):

                        position = str(acc_positions_response.json()["securitiesAccount"]["positions"][i]["instrument"]["description"])  # [0] bc only one position right now
                        position_option_symbol = acc_positions_response.json()["securitiesAccount"]["positions"][i]["instrument"]["symbol"]  # [0] bc only one position right now
                        position_symbol = acc_positions_response.json()["securitiesAccount"]["positions"][i]["instrument"]["underlyingSymbol"]  # [0] bc only one position right now
                        position_num_contracts = int(acc_positions_response.json()["securitiesAccount"]["positions"][i]["shortQuantity"])  # [0] bc only one position right now
                        position_avg_cost = round(float(acc_positions_response.json()["securitiesAccount"]["positions"][i]["averagePrice"]), 2)
                        maintence_req = acc_positions_response.json()["securitiesAccount"]["positions"][i]["maintenanceRequirement"] 
                        dist_to_max_profit = acc_positions_response.json()["securitiesAccount"]["positions"][i]["marketValue"] 
                        
                        option_quote_response = c.get_quote(position_option_symbol)
                        #print(json.dumps(option_quote_response.json(), indent=4))
                        position = str(option_quote_response.json()[position_option_symbol]["description"])  # confirms Description is correct, RSX was messing this up
                        price = option_quote_response.json()[position_option_symbol]["underlyingPrice"] 
                        dte = option_quote_response.json()[position_option_symbol]["daysToExpiration"]
                        tvol = round(float(option_quote_response.json()[position_option_symbol]["volatility"]) * np.sqrt(dte / 253), 3)
                        maxProfit = dist_to_max_profit * -1

                        # add to position_df so comparision to maxRR wont fail if position not in df

                        # Rewrite position description w/o 0 in front of day number if < 10
                        position = position.split()
                        if position[2][0] == '0':
                                position[2] = position[2][1]
                        strike = float(position[4])
                        position = " ".join(position[0:6])

                        breakeven = strike - position_avg_cost
                        pctToStrike = round((strike - price) * 100/ price, 2)
                        pctToBE = round((breakeven - price) * 100/ price, 2)


                        bid = float(option_quote_response.json()[position_option_symbol]["bidPrice"])
                        ask = float(option_quote_response.json()[position_option_symbol]["askPrice"])
                        mark = float(option_quote_response.json()[position_option_symbol]["mark"])

                        position_df_dict = {'Position': position, 'OptionSymbol': position_option_symbol, 'Symbol': position_symbol, 'DTE': dte, 'Strike': strike,
                                        'Price': price, 'AvgCost': position_avg_cost, 'Breakeven': breakeven, 'Contracts': position_num_contracts,
                                        'MaxProfit': maxProfit, 'Bid': bid, 'Ask': ask, 'Mark': mark, 'PctToStrike': pctToStrike,
                                        'PctToBE': pctToBE, 'TVol': tvol}

                        position_df = position_df.append(position_df_dict, ignore_index = True)

                position_df = MonteCarlo(position_df)

        except KeyError:
                print('KeyError Encounted (No Position)')

        # For No Intra Day Change or Updates
        # portfolio_value = acc_response.json()["securitiesAccount"]["initialBalances"]["accountValue"]
        # buying_power = acc_response.json()["securitiesAccount"]["initialBalances"]["buyingPower"]

        # Changes and updates Intra Day
        portfolio_value = acc_response.json()["securitiesAccount"]["currentBalances"]["liquidationValue"]
        buying_power = acc_response.json()["securitiesAccount"]["currentBalances"]["buyingPower"]
        cash_balance = acc_response.json()["securitiesAccount"]["currentBalances"]["cashBalance"]
        day_trades = acc_positions_response.json()["securitiesAccount"]["roundTrips"]


        main_call = acc_response.json()["securitiesAccount"]["projectedBalances"]["maintenanceCall"]

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Portfolio Value:  ${}".format(portfolio_value))
        print("Buying Power:     ${}".format(buying_power))
        #print("Current Position: {},  {} contracts @ ${},  {} from max profit".format(position,position_num_contracts,position_avg_cost,dist_to_max_profit))


        #acc_column_names = ['PortfolioValue','BuyingPower','CashBalance','MainCall']
        #account_df = pd.DataFrame(columns=acc_column_names)
        #account_df = {'PortfolioValue': portfolio_value,'BuyingPower': buying_power,'CashBalance': cash_balance,'MainCall': main_call*-1}
        
        # num_contracts_to_satisfy = margin_call / (current_position_strike*100)
        # slice 'symbol' to get number (strike) after the 'P' (for Put) 

        return portfolio_value, buying_power, cash_balance, main_call*-1, position_df


def get_watchlist():

        watchlistId = 1
        blackout_list = ["UVXY","RSX","VIXY","LABD","SQQQ","SOXS","VXX"]
        symbol_list = []

        wl_response = c.get_watchlists_for_single_account(config.account_id)
        #print(json.dumps(wl_response.json(), indent=4))
        for i in range(len(wl_response.json())):
                if wl_response.json()[i]["name"] == "MA Scan 2-3":
                        watchlistId = wl_response.json()[i]["watchlistId"]

        if watchlistId == 1:
                symbol_list = pd.read_csv("/Users/jonflees/FLEES-ANALYTICS-LOCAL/tdameritrade/MA-Scanner.csv", skiprows=3, usecols=["Symbol"]).values.flatten().tolist()

        else:
                wl2_response = c.get_watchlist(config.account_id, watchlist_id=watchlistId)
                for i in range(len(wl2_response.json()["watchlistItems"])):
                        if wl2_response.json()["watchlistItems"][i]["instrument"]["symbol"] not in blackout_list:
                                symbol_list.append(wl2_response.json()["watchlistItems"][i]["instrument"]["symbol"])

        return symbol_list



def sendSMS(maxRR, keep_or_switch):

        # Sending SMS of best option
        from twilio.rest import Client
        account_sid = "AC222ad863494c100a0bdb71e1866b6e63"
        auth_token = 'f0271094a69912757380b87591a5d3f5'
        client = Client(account_sid, auth_token)

        message_string = "{}\n\nCurrent Price: ${}\n\nBid: ${}\n\nBreakeven: ${}\n\nContracts: {}\n\nMax Profit: ${}\n\nExpected Profit: ${}\n\nPctStrike: {}%\n\nPctBE: {}%\n\nProbOTM: {}%\n\nProbBE: {}%\n\nRR Score: {}".format(
                                maxRR[15],maxRR[3],maxRR[4],maxRR[5],maxRR[6],maxRR[7],maxRR[18],maxRR[9],maxRR[10],maxRR[16],maxRR[17],maxRR[19])

        message3 = client.messages .create(
                        body =  "{}!\n\n{}".format(keep_or_switch, message_string), #Message you send
                        from_ = "+18454151297", #Provided phone number
                        to =    "+16572821212") #Your phone number
        message3.sid


def sendMarginCallSMS(main_call, num_buyback_to_satisfy):

        # Sending SMS of margin call alert
        from twilio.rest import Client
        account_sid = "AC222ad863494c100a0bdb71e1866b6e63"
        auth_token = 'f0271094a69912757380b87591a5d3f5'
        client = Client(account_sid, auth_token)

        message_string = "You have a maintenance call of ${}\n\nSatisfy with buying back {} contracts".format(main_call, num_buyback_to_satisfy)

        message3 = client.messages .create(
                        body =  "ATTENTION!\n\n" + message_string, #Message you send
                        from_ = "+18454151297", #Provided phone number
                        to =    "+16572821212") #Your phone number
        message3.sid


def rrToCSV(df, position_df):
        
        # Only strikes < 15% gain for PctToStrike, sorted by RR
        good_strikes = df.loc[df['Strike'] <= df['Price']*1.2].sort_values(by=['RR1.0'], ascending=False)

        good_strikes = good_strikes.loc[good_strikes['Bid'] > 0]

        good_strikes = good_strikes.dropna(axis=0)

        good_strikes.to_csv('Stonks_1st.csv')
        print("Stonks_1st.csv updated")

        good_strikes_200 = good_strikes.nlargest(200, 'RR1.0')

        great_strikes = MonteCarlo(good_strikes_200)

        great_strikes = great_strikes.loc[great_strikes['RR2.0'] > 0.001].sort_values(by=['RR2.0'], ascending=False)

        great_strikes.to_csv('Stonks.csv')
        print("Stonks.csv updated")

        # Get and store the highest RR score option as a list
        maxRR = great_strikes.loc[great_strikes['RR2.0'] == max(great_strikes['RR2.0'])].values.flatten().tolist()
        # print(maxRR)
        return maxRR, great_strikes


def compareCurrentPositionToMaxRR(maxRR, position_df, cash_balance, portfolio_value):

        
        pos_in_df = position_df.values.flatten().tolist()
        print("\nEvaluating switching positions...\n")
        keep_or_switch = ""

        # Check if position exists
        if len(pos_in_df) > 0:

                print("Current Position's RR: {}".format(pos_in_df[19]))

                # Compare RR Scores
                if pos_in_df[19] < maxRR[19]:

                        # Cost to switch
                        proj_acc_value = cash_balance - (pos_in_df[11] * pos_in_df[8] * 100)
                        upd_contracts = proj_acc_value / maxRR[2] / 100   # update num contracts, projected account value divided by strike
                        ratio = upd_contracts / maxRR[6]                # previous num contracts / upd num contracts
                        
                        print('Ratio: {:.4f}'.format(ratio))
                        print('Projected Account Value: {}'.format(proj_acc_value))

                        new_rr = get_risk_reward(maxRR[7]*ratio, maxRR[18]*ratio, maxRR[16], maxRR[17], maxRR[1], maxRR[10])
                        print('New RR Score: {}\n'.format(new_rr))

                        # Set DTE = 1 for calculations if its 0dte
                        maxRR[1] = maxRR[1] if maxRR[1] > 0 else 1
                        pos_in_df[3] = pos_in_df[3] if pos_in_df[3] > 0 else 1

                        if ( (maxRR[17]*ratio*maxRR[18]/100/maxRR[1]) + proj_acc_value) > ( (pos_in_df[17]*pos_in_df[18]/100/pos_in_df[3]) + portfolio_value):
                                print("Sell & switch positions\n")
                                keep_or_switch = "SWITCH"

                                # This is where to implement the selling then buying orders to switch positions

                        else:
                                print("Keep current position\n")
                                keep_or_switch = "KEEP"
                else:
                        print("Keep current position\n")
                        keep_or_switch = "KEEP"
        else:
                print("No current position\n")
                keep_or_switch = "SWITCH"

        return keep_or_switch


def AddStrikeToMySQL(maxRR):
        mycursor.execute(
        '''
        INSERT INTO Stonks (TimeStamp, Description, Symbol, DTE, Strike, Price, Bid, Breakeven, Contracts, MaxProfit,
                            MaxLoss, PctToStrike, PctToBE, Delta, TVol, ProfitDTE, RR1, ProbOTM, ProbBE, ExpProfit, RR2) 
        VALUES (CURRENT_TIMESTAMP(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', (maxRR[15],maxRR[0],maxRR[1],maxRR[2],maxRR[3],maxRR[4],maxRR[5],maxRR[6],maxRR[7],maxRR[8],
                maxRR[9],maxRR[10],maxRR[11],maxRR[12],maxRR[13],maxRR[14],maxRR[16],maxRR[17],maxRR[18],maxRR[19]))
        db.commit()


def deleteRow(description):
        mycursor.execute(
                '''
                DELETE FROM Stonks WHERE (Description = '{}')
                '''.format(description))
        db.commit()


def getLastMaxRR():
        mycursor.execute(
                '''
                SELECT Description FROM Stonks WHERE optionID=(SELECT MAX(optionID) FROM Stonks)
                ''')
        last_maxRR = mycursor.fetchone()
        print("Last MaxRR:", last_maxRR[0])
        return last_maxRR[0]


def RunAlgo():
        # Time Start of Algo
        t = time.time()
        start_time = datetime.datetime.now().ctime()
        print("\nAlgo Start:       {} PST".format(start_time[:16])) 

        portfolio_value, buying_power, cash_balance, main_call, position_df = get_account_info()
        syms = get_watchlist()
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Watchlist:\n{}".format(syms) )

        df = get_all(syms, portfolio_value)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        maxRR, df2 = rrToCSV(df, position_df)

        keep_or_switch = compareCurrentPositionToMaxRR(maxRR, position_df, cash_balance, portfolio_value)

        last_maxRR = getLastMaxRR()

        AddStrikeToMySQL(maxRR)
        #deleteRow(maxRR[15])

        sendSMS(maxRR, keep_or_switch) if maxRR[15] != last_maxRR else print("Same last MaxRR\n")

        if main_call > 0:                                       # FIX THIS
                num_buyback_to_satisfy = round(main_call / (position_df['Strike']*100), 0) + 1   # Always add 1 to cushion
                buyback_cost = num_buyback_to_satisfy * position_df['Ask']
                sendMarginCallSMS(main_call, num_buyback_to_satisfy)

        avg_maxProfit = float(df2['MaxProfit'].mean())
        df_exROI = df2
        df_exROI['ExpROI'] = df_exROI['ExpProfit'] / df_exROI['MaxLoss'] * 100  # ACTUALLY IS ROI
        df_exROI = df_exROI.sort_values(by=['ExpROI'], ascending=False)

        fig = px.bar(df_exROI, x='Description', y='ExpROI', #width=df2['MaxProfit']/avg_maxProfit*10,
                        hover_data=['Description','Price','Bid','Breakeven','Contracts','ExpProfit','PctToStrike','PctToBE','ProbOTM','ProbBE','RR2.0'], color='ProbBE')
        title = "Stonks Breakdown as of {}".format(start_time[:16])
        fig.update_layout(title_text = title, title_x = 0.5)
        py.plot(fig, filename='Stonks_bar', auto_open=False)
        #fig.show()

        # Time End of Algo
        elapsed = time.time() - t
        mins = elapsed // 60
        secs = elapsed % 60
        print('Elapsed time:      {:.0f} min {:.0f} seconds'.format(mins,secs))
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
  

def CheckIfExpOTM():

        tz_NY = pytz.timezone('America/New_York') 
        today = datetime.datetime.now(tz_NY).date()

        mycursor.execute(
                '''
                SELECT * FROM Stonks
                ''')
        stonk_from_db = mycursor.fetchall()

        for row in range(len(stonk_from_db)):

                exp_otm = stonk_from_db[row][22] # 'ExpiredOTM' in column position 22

                if exp_otm == None:       # Only run if 'ExpiredOTM' has no value currently

                        desc = stonk_from_db[row][2].split()     # Description is column position 2
                        exp = " ".join(desc[1:4])
                        exp = datetime.datetime.strptime(exp, '%b %d %Y').date()
                        
                        if today > exp:   # Only run following code if exp_date has passed
                                        
                                symbol = desc[0]
                                strike = float(desc[4])
                                symbol_exp_price_response = c.get_price_history(symbol, period_type=client.Client.PriceHistory.PeriodType.YEAR_TO_DATE,
                                                                                frequency_type=client.Client.PriceHistory.FrequencyType.DAILY,
                                                                                frequency=client.Client.PriceHistory.Frequency.DAILY)

                                #print(json.dumps(symbol_exp_price_response.json(), indent=4))

                                candles = symbol_exp_price_response.json()["candles"]

                                for i in range(len(candles)):

                                        day = symbol_exp_price_response.json()["candles"][i]["datetime"]
                                        day = datetime.date.fromtimestamp(day/1e3)
                                        day = day.strftime("%b %-d %Y")

                                        price_at_exp = float(symbol_exp_price_response.json()["candles"][i]["close"]) if day == exp else 0.01

                                if price_at_exp > strike:              
                                        
                                        sql = '''
                                        UPDATE Stonks 
                                        SET ExpiredOTM = "YES"
                                        WHERE (Description = %s)
                                        '''

                                else:
                                        sql = '''
                                        UPDATE Stonks 
                                        SET ExpiredOTM = "NO"
                                        WHERE (Description = %s)
                                        '''
                                vals = (stonk_from_db[row][2],)
                                mycursor.execute(sql, vals)
                                db.commit()
        
# 08/09/22 Additions
def ReadStockQuotes(symbol):
        response  = c.get_quote(symbol)
        #print(json.dumps(response.json(), indent=4))
        current_price = str(response.json()[symbol]["regularMarketLastPrice"])
        print(current_price)
        # myobj = gtts.gTTS(text=current_price, lang=language, slow=False)
        # myobj.save("welcome.mp3")
        # playsound("welcome.mp3")


def GetOptionsMarks(symbol):
        exp_dates, today, next_friday = getExpDates()
        response = c.get_option_chain(symbol, strike_count=4, from_date=today, to_date=next_friday)
        #print(json.dumps(response.json(), indent=4))

        calls_df = pd.DataFrame(columns = ['expDate', 'strike', 'mark', 'proj'])
        puts_df = pd.DataFrame(columns = ['expDate', 'strike', 'mark', 'proj'])


        for j in range(len(exp_dates)):

                for k in response.json()["callExpDateMap"][exp_dates[j]]:
                        mark = response.json()["callExpDateMap"][exp_dates[j]][k][0]["mark"]
                        expDate = response.json()["callExpDateMap"][exp_dates[j]][k][0]["expirationDate"]
                        proj = float(k) + float(mark)
                        calls_df = calls_df.append({'expDate': expDate, 'strike': k, 'mark': mark, 'proj': proj}, 
                        ignore_index = True)

                for l in response.json()["putExpDateMap"][exp_dates[j]]:
                        mark = response.json()["putExpDateMap"][exp_dates[j]][l][0]["mark"]
                        expDate = response.json()["callExpDateMap"][exp_dates[j]][k][0]["expirationDate"]
                        proj = float(l) - float(mark)
                        puts_df = puts_df.append({'expDate': expDate, 'strike': l, 'mark': mark, 'proj': proj}, 
                        ignore_index = True)

        # print(df)
        return calls_df, puts_df


def MakePlot(symbol):
        from math import pi
        import pandas as pd
        from bokeh.plotting import figure, show
        from bokeh.models import HoverTool
        from bokeh.transform import factor_cmap
        from bokeh.palettes import Spectral5

        response = c.get_price_history_every_thirty_minutes(symbol)     #five_minutes(symbol)    #_minute(symbol)
        #print(json.dumps(response.json(), indent=4))
        df = response.json()['candles']

        df = pd.DataFrame(df, columns=['datetime', 'open', 'close', 'high', 'low', 'volume'])
        #df['x'] = df.reset_index().index
        #print(df)

        calls_df, puts_df = GetOptionsMarks(symbol)

        inc = df['close'] > df['open']
        dec = df['close'] < df['open']
        w = 1000000           #30,000 for minute chart

        TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

        TOOLTIPS=[
                ("Open", "$open"),
                ("High", "$high"),
                ("Low", "$low"),
                ("Close", "$close"),
                ("Volume", "$volume")
                ]

        HoverTool(
                tooltips=[
                        ( 'open',   '$@open{%0.2f}'   ),
                        ( 'close',  '$@close{%0.2f}'  ), # use @{ } for field names with spaces
                        ( 'high',   '$@high{%0.2f}'   ),
                        ( 'low',   '$@low{%0.2f}'     ),
                        ( 'volume', '@volume{0.00 a}'  ),
                ],

                formatters={
                        '@open'  : 'printf', # use 'datetime' formatter for '@date' field
                        '@close' : 'printf',   # use 'printf' formatter for '@{adj close}' field
                                                # use default 'numeral' formatter for other fields
                },

                # display a tooltip whenever the cursor is vertically in line with a glyph
                mode='vline'
                )

        title_text = "{} Candlestick".format(symbol)

        p = figure(x_axis_type="datetime", tools=TOOLS, tooltips=TOOLTIPS, height=700, sizing_mode="stretch_width", title = title_text)
        #p.xaxis.major_label_overrides = {i: datetime.date.fromtimestamp(date).strftime('%Y-%m-%d %H:%S') for i, date in enumerate(df.index)}
        p.xaxis.major_label_orientation = pi/4
        p.grid.grid_line_alpha=0.3

        p.segment(df['datetime'], df['high'], df['datetime'], df['low'], color="gainsboro")
        p.vbar(df['datetime'][inc], w, df['open'][inc], df['close'][inc], fill_color="mediumseagreen", line_color="darkgreen")
        p.vbar(df['datetime'][dec], w, df['open'][dec], df['close'][dec], fill_color="crimson", line_color="darkred")

        p.circle(calls_df['expDate'], calls_df['proj'], fill_color="mediumseagreen", line_color="darkgreen")
        p.circle(puts_df['expDate'], puts_df['proj'], fill_color="crimson", line_color="darkred")

        p.background_fill_color = (5, 5, 5)
        p.xgrid.grid_line_color = "slategray"
        p.xgrid.grid_line_dash = [6,4]
        p.ygrid.grid_line_color = "slategray"
        p.ygrid.grid_line_dash = [6,4]

        # cr = p.circle(x, y, size=20,
        #         fill_color="grey", hover_fill_color="firebrick",
        #         fill_alpha=0.05, hover_alpha=0.3,
        #         line_color=None, hover_line_color="white")

        #p.add_tools(HoverTool(tooltips=None, renderers=[cr], mode='hline'))

        show(p)



# ***** CAUTION ***** 
# SUCCESSFULLY PLACES ORDERS
# response = c.place_order(config.account_id, 
#                          equity_buy_limit('BBIG', 5, 3.0)
#                          .set_duration(Duration.GOOD_TILL_CANCEL)
#                          .set_session(Session.NORMAL)
#                          .build())
# print(response)  # <Response [201 ]> if it worked



# cross reference with earnings calender to see if getting an IV bump bc earnings

#plt.bar()


# 1. FORM LIST FROM STOCK SCREENER
# 2. SEND LIST THROUGH GET_ALL FUNCTION
# 3. FIX/PERFECT THE RR SCORE






# should probOTM be adjusted by macro market state? bull, bear, or sideways?

# probOTM
# probTouch

# add scanner
# must learn how to fine tune and get better results from tda scanner tho

# risk/reward value = (profit * prob) "expected return" / cost * (1-prob) "expected cost"

# calculate an expectation of closing early at .05 (profit - (.05*contract))

# utilize weighted coefficients for probOTM and probTouch? would make a better prob?
# the answer requires backtesting and ML
# typical prob plus stock's prob (IV?) to create newer specialized prob

# is picking best risk/reward value option the best route

# be careful not to train on too old of data... a progressive importance weighted factor based
# time from now?
# big events included and should be weighted higher bc we usually look back at those 
# as key points to try to predict future

# find best probability for predicting success in this price range over time
# and use machine learning to improve the system
# this prob will be what gets used in the riskreward calc

# dont want a top 1 but rather a top 3 for me to pick from

# get the system to recognize a losing trade and alert me to recommend exit with good timing

# maybe the best probability calculator already exists? can be improved?

# probability max profit, probability any profit

# only have to compare stocks in Sell Puts watchlist
# can add manually if needed

# sleep_minutes = 60*5   # 5 minutes

while True:

        start_time = datetime.datetime.now().ctime()

        if int(start_time[11:13]) >= 6 and int(start_time[11:13]) <= 13:  # program runs between 6am - 1pm PST
                
                #RunAlgo()
                ReadStockQuotes('TTCF')
                time.sleep(5)         # 60sec * 10min

        else:
                
                print("\nLoop Lap Time:    {} PST".format(start_time[:16]))
                # Check if options in DB were OTM @ exp
                #CheckIfExpOTM()
                #RunAlgo()
                #ReadStockQuotes('AAPL')
                MakePlot('AAPL')
                time.sleep(30)         # 60sec * 10min

# Send text of current positions rank


def add_candle_to_SQL(date_time, open, close, high, low, average, volume):

        # In case a duplicate where to get passed through it would just update
        mycursor.execute(
        '''
        INSERT INTO ARVL (DateTime, Open, Close, High, Low, Average, Volume) 
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE DateTime=DateTime; 
        ''', (date_time, open, close, high, low, average, volume))
        db.commit()


def get_candles_from_SQL(start_date):
        
        column_names = ['DateTime','Open','Close','High','Low','Average','Volume']
        candles_df = pd.DataFrame(columns=column_names, dtype=object)

        mycursor.execute("SELECT * FROM tda_master_database.ARVL WHERE DateTime >= {}".format(start_date))
        candles = mycursor.fetchall()
        
        for row in range(len(candles)):

                date_time,open,close,high,low,avg,volume = candles[row][0],float(candles[row][1]),float(candles[row][2]),float(candles[row][3]),float(candles[row][4]),float(candles[row][5]),float(candles[row][6])
                candles_df_dict = {'DateTime': date_time, 'Open': open, 'Close': close, 'High': high, 'Low': low,
                                        'Average': avg, 'Volume': volume}

                candles_df = candles_df.append(candles_df_dict, ignore_index = True)

        candles_df.to_csv('ARVL_candles.csv')
        return candles_df


def check_if_already_in_SQL():

        date_time_SQL_list = []
        mycursor.execute("SELECT DateTime FROM tda_master_database.ARVL")
        candles = mycursor.fetchall()
        for row in range(len(candles)):
                date_time_SQL_list.append(candles[row][0].strftime("%Y-%m-%d %H:%M:%S"))      # DateTime is in position 0
        
        return date_time_SQL_list


def get_price_history(symbol):

        symbol_price_history_response = c.get_price_history(symbol, period_type=client.Client.PriceHistory.PeriodType.DAY,
                                                                                frequency_type=client.Client.PriceHistory.FrequencyType.MINUTE,
                                                                                frequency=client.Client.PriceHistory.Frequency.EVERY_MINUTE,
                                                                                need_extended_hours_data=False)

        #print(json.dumps(symbol_price_history_response.json(), indent=4))

        candles = symbol_price_history_response.json()["candles"]

        for i in range(len(candles)):

                date_time = symbol_price_history_response.json()["candles"][i]["datetime"]
                date_time = datetime.datetime.fromtimestamp(date_time/1e3)
                date_time = date_time.strftime("%Y-%m-%d %H:%M:%S")

                date_time_SQL_list = check_if_already_in_SQL()

                if date_time not in date_time_SQL_list:
                        open = symbol_price_history_response.json()["candles"][i]["open"]
                        close = symbol_price_history_response.json()["candles"][i]["close"]
                        high = symbol_price_history_response.json()["candles"][i]["high"]
                        low = symbol_price_history_response.json()["candles"][i]["low"]
                        volume = symbol_price_history_response.json()["candles"][i]["volume"]
                        avg = round((high+low+open+close) / 4, 4)
                        
                        print("Day Time: {}\nOpen: {}\nClose: {}\nHigh: {}\nLow: {}\nAverage: {}\n".format(date_time, open, close,
                                                                                                                high, low, avg))
                        add_candle_to_SQL(date_time, open, close, high, low, avg, volume)







### WHATEVER I WAS TESTING BEFORE

# # get_price_history('ARVL')

# candles_df = get_candles_from_SQL('2022-03-07')
# print(candles_df)

# averages = candles_df['Average'].values.flatten().tolist()
# averages = np.asarray(averages)

# # Find peaks(max).
# peak_indexes = signal.argrelextrema(averages, np.greater)
# peak_indexes = peak_indexes[0]
 
# # Find valleys(min).
# valley_indexes = signal.argrelextrema(averages, np.less)
# valley_indexes = valley_indexes[0]

# # date_time_SQL_list = check_if_already_in_SQL()

# # print(date_time_SQL_list)

# # # Plot main graph.
# # fig = make_subplots(specs=[[{"secondary_y": True}]])
 
# # # Plot peaks.
# # peak_x = peak_indexes
# # peak_y = averages[peak_indexes]
# # trace1 = go.Scatter(peak_x, peak_y) #, marker='o', linestyle='dashed', color='green', label="Peaks")
 
# # # Plot valleys.
# # valley_x = valley_indexes
# # valley_y = averages[valley_indexes]
# # trace2 = go.Scatter(valley_x, valley_y) #, marker='o', linestyle='dashed', color='red', label="Valleys")

# # trace_og = go.Line(candles_df, x='DateTime', y='Average',
# #                         hover_data=['DateTime','Open','Close','High','Low','Average','Volume'])

# # fig.add_trace(trace_og)
# # fig.add_trace(trace1,secondary_y=True)
# # fig.add_trace(trace2,secondary_y=True)
# fig = go.Figure(go.Candlestick(x=candles_df['DateTime'],
#                                 open=candles_df['Open'], high=candles_df['High'],
#                                 low=candles_df['Low'], close=candles_df['Close']))
# title = "ARVL Chart"
# fig.update_layout(title_text = title, title_x = 0.5)
# py.plot(fig, filename='ARVL_chart', auto_open=False)

# # Plot main graph.
# (fig, ax) = plt.subplots()
# ax.plot(candles_df.index, averages)
 
# # Plot peaks.
# peak_x = peak_indexes
# peak_y = averages[peak_indexes]
# ax.plot(peak_x, peak_y, marker='o', linestyle='dashed', color='green', label="Peaks")
 
# # Plot valleys.
# valley_x = valley_indexes
# valley_y = averages[valley_indexes]
# ax.plot(valley_x, valley_y, marker='o', linestyle='dashed', color='red', label="Valleys")

# plt.show()


# Find stocks that have high value options to sell
# Gather historical data and back test before ever letting algo run a trade
# Sell puts/calls for that stock