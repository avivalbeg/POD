from bot.tools.mongo_manager import GameLogger
from util import encodeRec, DummyLogger

"""
Script for downloading zipped irc data into a "data" directory.
"""

from bs4 import BeautifulSoup
import requests
import os 
from os.path import join, exists
from constants import IRC_DATA_PATH, POKER_BOT_DATA_PATH, GameStages
from copy import copy
import sys
from bot.decisionmaker.montecarlo_python import MonteCarlo
import time
import re
from pprint import pprint
import pandas as pd



def find_files(url):

    soup = BeautifulSoup(requests.get(url).text)

    for a in soup.find_all('a'):
        yield a['href']

def getIrcData():
    if not exists("data"):
        os.mkdir("data")
    
    if not exists(IRC_DATA_PATH):
        os.mkdir(IRC_DATA_PATH)
    
    url = 'https://web.archive.org/web/20051029232106/http://games.cs.ualberta.ca:80/poker/IRC/IRCdata/'
    
    links = []
    for link in find_files(url):
        links.append(link)
        
    links = links[4:]
    image_url = links[0]
    for elem in links[1:]:
        r = requests.get('https://web.archive.org/web/20051029232106/http://games.cs.ualberta.ca:80/poker/IRC/IRCdata/' + elem)  # create HTTP response object
        with open(join("data", elem), 'wb') as encodeRec:
            encodeRec.write(r.content)
            encodeRec.close()


def getEquity(mycards, tableCards,nActivePlayers):
    tup = ((tuple(sorted(mycards)),),
           tuple(sorted(tableCards)),
           nActivePlayers)
    
    mc = MonteCarlo()
    timeout = time.time() + 5
    mc.run_montecarlo(DummyLogger(),
                      [mycards],
                      tableCards,
                      player_amount=len(nActivePlayers),
                      ui=None,
                      timeout=timeout,
                      maxRuns=10000,
                      ghost_cards="",
                      opponent_range=0.25)
    return mc.equity

unimportantFeatures = [
    "logger",
    "logging_timestamp",
    "ip",
    "version",
    "computername",
    "_id",
    "GameID",
    "uploader",
    
    ]


def cleanMdbDic(dic):
    
    cleaned = {}
    for field, val in dic.items():
        
        if type(val)==type(b""):
            val=val.decode(sys.stdout.encoding)
        
        if val=="False": val = 0
        if val=="True": val = 1
        if val=="nan" or not val: val = -1
        if val in GameStages: val = GameStages.index(val)
        
        if field == "other_players": val=str(val) 
        if field== "PlayerCardList": val = " ".join(eval(val))
        if field== "PlayerCardList_and_others": val = " ".join(re.findall("[a-zA-Z\d]+", val))
        cleaned[field] = [val]
        
    return cleaned 

def dlPokerBotData(): 
    """
    Download and save the data stored on the poker bot's server.
    Each file corresponds to a game, and each column of a game is a roundData.
    I only found games with up to 4 rounds, so maybe each roundData is 
    actually a game stage? 
    """
    
    if not exists("data"):
        os.mkdir("data")
    if not exists(POKER_BOT_DATA_PATH):
        os.mkdir(POKER_BOT_DATA_PATH)
        
    gameLogger = GameLogger()   
    rounds = gameLogger.get_neural_training_data()

    
    
    for roundData in rounds:
        # Create a dataframe from roundData
        d = encodeRec(roundData)
        vals = d["round_values"]
        roundDf = pd.DataFrame(cleanMdbDic(vals))
        thisGameId = roundDf["GameID"][0]
        roundDf= roundDf.transpose()
        if thisGameId != prevGameId:
            
            # Save previous frame
            if type(gameDf) != type(None):
                gameDf.to_csv(join(POKER_BOT_DATA_PATH, prevGameId+".csv"))

            # Start collecting rounds as columns
            gameDf = roundDf.copy()
            prevGameId = copy(thisGameId)
        else: 
            # Append current roundData to game
            gameDf = pd.concat((gameDf, roundDf), axis=1)
