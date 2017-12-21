"""
This module defines a class that harvests useful data out of logged poker games.
Currently it can only harvest and log hand data to predict player equity based 
on game state. 
"""


import sys, os
sys.path = [os.getcwd()] + sys.path

import matplotlib
import pandas as pd
import time
import numpy as np
from mining.data_parsing import IrcHoldemDataParser
from bot.table_analysers.base import Table
from bot.decisionmaker.montecarlo_python import MonteCarlo
from copy import copy
from bot.decisionmaker.current_hand_memory import History, \
    CurrentHandPreflopState
from bot.decisionmaker.decisionmaker import Decision
from bot.table_analysers.table_screen_based import TableScreenBased
from random import choice

import os, sys
from bot.tests import init_table
from bson.json_util import default
from pprint import pprint
from bot.tools.mongo_manager import StrategyHandler, GameLogger
from os.path import join, exists
# os.environ['KERAS_BACKEND']='theano'

import logging.handlers
import pytesseract
import threading
import datetime
from PIL import Image, ImageGrab
from PyQt5 import QtWidgets, QtGui
from configobj import ConfigObj

from enum import Enum
from constants import *
from util import *


# Remove duplicates
FEATURES = [
    "h.round_number",
    "table.myPos",
    "GameStages.index(table.gameStage)",
    "table.first_raiser",
    "table.second_raiser",
    "table.first_caller",
    "table.second_caller",
    "table.round_pot_value",
    "table.currentCallValue",
    "table.currentBetValue",
    "table.global_equity",  # Without any hand; how good are the cards on the table
    "len(table.other_players)",
    "len(table.other_active_players)",
    "table.totalPotValue",
    "table.max_X",
    "table.myFunds",
    "h.myLastBet",
    "table.nMyCalls",
    "table.nMyRaises", # Includes blinds, bets and raises
    "table.myCallSum",
    "table.myRaiseSum", # Includes blinds, bets and raises
    "table.currentBetValue", # I dont understand the difference between current bet value and current call value,
    "table.currentCallValue", # so I treat them the same for now
    ]

# These features will not be turned into integers
DOUBLE_FEATS = [
    FEATURES.index("table.global_equity"),
    ]

class DummyTable:
    pass
class DummyHistory:
    pass

class IrcDataMiner:
    
    def __init__(self, ircDataPath=IRC_DATA_PATH, handDataPath=HAND_DATA_PATH, debugMode=False):
        self._debugMode = debugMode
        self._ircParser = IrcHoldemDataParser(ircDataPath)
        # Counters that get set back to 0 every stage        
        self._stageCallCounter = 0  # How many calls so far in this stage
        self._stageRaiseCounter = 0  # How many raises so far in this stage
        self._stageBetCounter = 0  # How many bets so far in this stage
        if not exists(handDataPath):
            os.mkdir(handDataPath)
        
        self._handDataPath = handDataPath
        self._fileCounter = 1
        self._relEquityCache = {}
        self._globalEquityCache = {}
        
    def mineHandData(self, debugMode=False):
        """
        Collects featurized games for players that made it to showdown. Stores data in text vector format. 
        Each generated file can be loaded with:
            >>pandas.DataFrame.from_csv(<file_name>, sep=" ")
        """
        self._debugMode = debugMode
        print("Creating file #%d" % self._fileCounter)
        open(join(self._handDataPath, "%d.txt" % self._fileCounter), "w").close()
        i = 0
        self._outFile = open(join(self._handDataPath, "%d.txt" % self._fileCounter), "a")
        for game in self._ircParser.iterGames():
            i += 1
            # Choose game and go over all players with cards            
            game = self._ircParser.nextGame()
            players = [player for player in game.players if player.cards]
            for player in players:
                self.collectDataFromGame(game, player)
            
            # Create a new file
            if i % 100 == 0:
                self._outFile.close()
                self._fileCounter += 1
                print("Creating file #%d" % self._fileCounter)
                open(join(self._handDataPath, "%d.txt" % self._fileCounter), "w").close()
                self._outFile = open(join(self._handDataPath, "%d.txt" % self._fileCounter), "a")
        self._outFile.close()
        
    def collectDataFromGame(self, game, player, retAt=None):
        h = History()
        table = self.newTable(game, player, h)
        
        if self._debugMode:
            print (game)
        # Go through the game and save all data
        while table.gameStage != Showdown:
            self.playStage(game, player, table, h, retAt)
        
    
    def playStage(self, game, player, table, h, retAt=None):
        if table.gameStage != PreFlop:
            table.max_X = 1
        someonePlayed = True  # Tells us if there are still players who didn't quit/fold


        
        table.round_pot_value = 0
        while someonePlayed:
            someonePlayed = self.playRound(game, player, table, h)
            if (table.gameStage, h.round_number) == retAt:
                return self.featurizeTable(table, h, player)
            h.round_number += 1

        # Re-init
        h.round_number = 0 
        table.currentBetValue = 0
        table.currentCallValue = 0

        
        table.gameStage = getOrDefault(GameStages,
                                       GameStages.index(table.gameStage) + 1,
                                       None)
        
        
        # Expose cards
        if table.gameStage == Flop:
            table.cardsOnTable = list(map(ircCardToBotCard, game.boardCards[:3]))
        if table.gameStage == Turn:
            table.cardsOnTable = list(map(ircCardToBotCard, game.boardCards[:4]))
        if table.gameStage == River:
            table.cardsOnTable = list(map(ircCardToBotCard, game.boardCards))
        
        table.global_equity = self.getAbsEquity(table)
        table.equity = self.getEquity(table)
        
    def playRound(self, game, player, table, h):
        
        nextStage = GameStages[GameStages.index(table.gameStage) + 1]
        prevStage = GameStages[GameStages.index(table.gameStage) + 1]
        
        # global bet: divide money from this stage by number of players that put in money this turn
        betAvgGlobal = divOr0((getattr(game, nextStage.lower() + "Pot") - getattr(game, table.gameStage.lower() + "Pot"))\
                    , len(list(filter(lambda plr: re.findall("[BrbcA]",
                                     getattr(plr, table.gameStage.lower() + "Actions"))
                                      , game.players))))
        
        someonePlayed = False

        for curPlayer in game.players:
            
            # Find action
            actions = getattr(curPlayer, table.gameStage.lower() + "Actions")
            action = getOrDefault(actions,
                                  h.round_number,
                                  NA)
            
            if action == RAISE:
                if np.isnan(table.first_raiser):
                    table.first_raiser = curPlayer.pos
                elif (not np.isnan(table.first_raiser)) \
                    and np.isnan(table.second_raiser):
                        table.second_raiser = curPlayer.pos
            if action == CALL:
                if np.isnan(table.first_caller):
                    table.first_caller = curPlayer.pos
                elif (not np.isnan(table.first_caller)) \
                    and np.isnan(table.second_caller):
                        table.second_caller = curPlayer.pos
                
            # Weigh bet by number of money-placing actions this stage    
            betAvg = divOr0(betAvgGlobal , len(re.findall("[AbrcB]", actions))) 
            
            # If even one player did something, 
            # this game stage continues
            if not action in (NA, QUIT, KICKED):
                someonePlayed = True
                if self._debugMode:
                    print(curPlayer.name, action, int(betAvg), int(table.totalPotValue), table.gameStage, h.round_number)
            
            # Update table based on action
            if action == CALL:
                if curPlayer == player:
                    table.myFunds -= betAvg
                    table.nMyCalls += 1
                    table.myCallSum += betAvg
                table.totalPotValue += betAvg
                self._stageCallCounter += 1
                table.round_pot_value += betAvg
            
            # I still haven't handled all in
            elif action in (BLIND, BET, RAISE, ALL_IN):
                if curPlayer == player:
                    h.myLastBet = betAvg
                    table.myFunds -= betAvg
                    table.nMyRaises += 1
                    table.myRaiseSum += betAvg

                table.currentBetValue = betAvg
                table.currentCallValue = betAvg
                table.totalPotValue += betAvg
                table.round_pot_value += betAvg
                
            elif action == CHECK:
                if curPlayer == player:
                    table.nMyChecks += 1
            
            elif action == FOLD:
                table.other_active_players = table.other_active_players.difference({curPlayer.name})
            elif action == KICKED:
                table.other_active_players = table.other_active_players.difference({curPlayer.name})
            elif action == QUIT:
                table.other_active_players = table.other_active_players.difference({curPlayer.name})
                
            elif action == NA:
                pass
            
            else:
                raise ValueError("Unexpected action " + action)
            
            
            # Note: These features represent the game state
            # AT THE END OF THIS ROUND. So the first feature
            # dictionary that is returned is after the 0th (or first) 
            # round of the preflop
            if curPlayer == player and action != NA:
                features = self.featurizeTable(table, h, player)
                
                if self._debugMode:
                    pprint({FEATURES[i]:features[i] for i in range(len(FEATURES))})
                # Comment this out if you're doing unittesting
                self._outFile.write(" ".join([str(x) for x in features]) + " " + str(table.equity) + "\n")
            
        return someonePlayed

    def getEquity(self, table):
        
        tup = ((tuple(sorted(table.mycards)),),
               tuple(sorted(table.cardsOnTable)),
               len(table.other_active_players))
        
        if tup in self._relEquityCache.keys():
            return self._relEquityCache[tup]
        mc = MonteCarlo()
        timeout = time.time() + 5
        mc.run_montecarlo(logging,
                          [table.mycards],
                          table.cardsOnTable,
                          player_amount=len(table.other_active_players),
                          ui=None,
                          timeout=timeout,
                          maxRuns=10000,
                          ghost_cards="",
                          opponent_range=0.25)
        
        self._relEquityCache[tup] = mc.equity
        return mc.equity  

    def getAbsEquity(self, table):
        tup = (tuple(sorted(table.cardsOnTable)), len(table.other_active_players))
        if tup in self._globalEquityCache.keys():
            return self._globalEquityCache[tup]
        mc = MonteCarlo()
        timeout = time.time() + 5
        mc.run_montecarlo(logging,
                          [table.cardsOnTable[:2]],
                          table.cardsOnTable[2:],
                          player_amount=len(table.other_active_players),
                          ui=None,
                          maxRuns=10000,
                          timeout=timeout,
                          ghost_cards="",
                          opponent_range=0.25)
        
        # Cache and return
        self._globalEquityCache[tup] = mc.equity
        return mc.equity  
    
    

    def featurizeTable(self, table, h, player):
        features = [
            h.round_number,
            table.myPos,
            GameStages.index(table.gameStage),
            table.first_raiser,
            table.second_raiser,
            table.first_caller,
            table.second_caller,
            table.round_pot_value,
            table.currentCallValue,
            table.currentBetValue,
            table.global_equity,  # Without any hand
            len(table.other_players),
            len(table.other_active_players),
            table.totalPotValue,
            table.max_X,
            table.myFunds,
            h.myLastBet,
            table.nMyCalls,
            table.nMyRaises,
            table.myCallSum,
            table.myRaiseSum,
            table.currentBetValue,
            table.currentCallValue,
            ]
        new = []
        for f in features:
            if np.isnan(f):
                new.append(-1)
            elif features.index(f) in DOUBLE_FEATS:
                new.append(f)
            else:
                new.append(int(f))
        return np.array(new)

    def newTable(self, game, player, h):
        
        # Init history (we don't really need it)
        h.round_number = 0
        preflop_url = "bot/decisionmaker/preflop.xlsx"  # This path is relative to the POD directory; if you're running it from elsewhere you might have to adjust it
        h.preflop_sheet_name = "preflop.xlsx"
        h.preflop_sheet = pd.read_excel(preflop_url, sheetname=None)
        h.myLastBet = 0

        # Init table 
        table = DummyTable()
        table.myPos = player.pos
        table.gameStage = PreFlop
        table.other_active_players = set([plyr.name for plyr in game.players if plyr != player])
        
        table.equity = 0
        table.global_equity = 0
        table.nMyRaises = 0
        table.nMyCalls = 0
        table.nMyChecks = 0
        table.myRaiseSum = 0
        table.myCallSum = 0

        table.first_raiser = np.nan
        table.second_raiser = np.nan
        table.first_caller = np.nan
        table.second_caller = np.nan
        table.first_raiser_utg = np.nan
        table.first_caller_utg = np.nan
        table.second_raiser_utg = np.nan
        
        table.mycards = [ircCardToBotCard(card) for card in player.cards]
        table.other_players = [othrPlyr for othrPlyr in game.players if othrPlyr.name != player.name]
        table.round_pot_value = 0
        table.currentCallValue = 0
        table.currentBetValue = 0
        table.relative_equity = 0
        table.global_equity = 0
        table.cardsOnTable = []
        table.totalPotValue = 0
        table.max_X = 0.86
        table.myFunds = player.bankroll

        
        return table


