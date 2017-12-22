"""Script for managing downloading, saving and scraping data.
The actual work isn't done here, rather, these are scripts
that manage objects that do the actual work."""



from bot.tools.mongo_manager import GameLogger
from util import encodeRec, DummyLogger

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
import numpy as np
from pprint import pprint
import pandas as pd
from mining.data_parsing import IrcHoldemDataParser
from constants import *


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


def getEquity(mycards, tableCards, nActivePlayers):
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

featsToIgnore = [
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
        
        if type(val) == type(b""):
            val = val.decode(sys.stdout.encoding)
        
        if val == "False": val = 0
        if val == "True": val = 1
        if val == "nan" or not val: val = -1
        if val in GameStages: val = GameStages.index(val)
        
        if field == "other_players": val = str(val) 
        if field == "PlayerCardList": val = " ".join(eval(val))
        if field == "PlayerCardList_and_others": val = " ".join(re.findall("[a-zA-Z\d]+", val))
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
    games = gameLogger.mongodb.games.find()
    
    prevRoundId = (np.nan, np.nan, np.nan)
    gameDf = np.nan
    
    for game in games:
        rounds = sorted(game["rounds"], key=lambda x:(x["round_values"]["gameStage"], x["round_number"]))
        if not rounds: continue
        gameDf = pd.DataFrame(cleanMdbDic(encodeRec(rounds[0]["round_values"])))
        for roundData in rounds[1:]:
            roundDf = pd.DataFrame(cleanMdbDic(encodeRec(roundData["round_values"])))
            thisRoundId = (roundDf["GameID"][0],
                           roundDf["gameStage"][0],
                           eval(roundDf["round_number"][0]))
            
            # Append current roundData to game
            gameDf = pd.concat((gameDf, roundDf), axis=0)
            prevRoundId = copy(thisRoundId)
        if type(gameDf)!=type(np.isnan):
            blacklist = ("rounds","GameID","ComputerName","Template","FinalDecision","_id","ip","logging_timestamp")
            fileName = game["GameID"]+"-"+"-".join([str(v) for k,v in game.items() if not k in blacklist])
            
            gameDf.transpose().to_csv(join(POKER_BOT_DATA_PATH, fileName+".csv"))



def mineGameData(ircDataPath=IRC_DATA_PATH, 
                 gameDataPath=GAME_VECTORS_PATH, 
                 debugMode=False):
    """Collect and store vector representation of 
    game rounds from the IRC dataset from the 
    perspective of players that made it to showdown."""
    
    ircParser = IrcHoldemDataParser(ircDataPath)
    
    # Counters that get set back to 0 every stage        
    _stageCallCounter = 0  # How many calls so far in this stage
    _stageRaiseCounter = 0  # How many raises so far in this stage
    _stageBetCounter = 0  # How many bets so far in this stage
    if not exists(gameDataPath):
        os.mkdir(gameDataPath)
    
    fileCounter = 1
    

    
    print("Creating file #%d" % fileCounter)
    open(join(gameDataPath, "%d.csv" % fileCounter), "w").close()
    i = 0
    outFile = open(join(gameDataPath, "%d.txt" % fileCounter), "a")
    for game in ircParser.iterGames():
        if debugMode:
            print(game)
        
        i += 1
        # Choose game and go over all players with cards            
        game = ircParser.nextGame()
        players = [player for player in game.players if player.cards]
        for player in players:
            for vector in game.roundVectors(player, debugMode):
                print(vector)
                quit()
                outFile.write("\t".join([str(x) for x in vector]))
            
        # Create a new file
        if i % 500 == 0:
            outFile.close()
            fileCounter += 1
            print("Creating file #%d" % fileCounter)
            open(join(gameDataPath, "%d.txt" % fileCounter), "w").close()
            outFile = open(join(gameDataPath, "%d.csv" % fileCounter), "a")

    outFile.close()        
    
    