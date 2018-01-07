


from pickle import load
from pprint import pprint
from unittest import TestCase

from bot.decisionmaker.current_hand_memory import History
from data_parsing import IrcHoldemDataParser
from DataMiner import IrcDataMiner, FEATURES
from constants import IRC_DATA_PATH, HAND_DATA_PATH
from os.path import join

class TestDataCollection(TestCase):

    dc = IrcDataMiner(join("..",IRC_DATA_PATH),join("..", HAND_DATA_PATH))
        
    def test1(self):
        with open("game-pickles/game-2.pkl", "rb") as encodeRec:
            game = load(encodeRec)
               
        player = game.players[1]
        featsDict = self.getFeatures(game, player, ("PreFlop", 1))
 
        self.assertEquals(featsDict['GameStages.index(table.gameStage)'], 0)
        self.assertEquals(featsDict['h.myLastBet'], 20)
        self.assertEquals(featsDict['h.round_number'], 1)
        self.assertEquals(featsDict['len(table.other_active_players)'], 2)
        self.assertEquals(featsDict['len(table.other_players)'], 3)
        self.assertEquals(featsDict['table.currentBetValue'], 20)
        self.assertEquals(featsDict['table.currentCallValue'], 20)
        self.assertEquals(featsDict['table.first_caller'], 4)
        self.assertEquals(featsDict['table.first_raiser'], -1)
        self.assertEquals(featsDict['table.global_equity'], 0)
        self.assertEquals(featsDict['table.max_X'], 0)
        self.assertEquals(featsDict['table.myCallSum'], 0)
        self.assertEquals(featsDict['table.myFunds'], 2236)
        self.assertEquals(featsDict['table.myPos'], 2)
        self.assertEquals(featsDict['table.myRaiseSum'], 20)
        self.assertEquals(featsDict['table.nMyCalls'], 0)
        self.assertEquals(featsDict['table.nMyRaises'], 1)
        self.assertEquals(featsDict['table.round_pot_value'], 60)
        self.assertEquals(featsDict['table.second_caller'], 1)
        self.assertEquals(featsDict['table.second_raiser'], -1)
        self.assertEquals(featsDict['table.totalPotValue'], 60)

    
    def test2(self):
        with open("game-pickles/game-2.pkl", "rb") as encodeRec:
            game = load(encodeRec)
        player = game.players[3]
        featsDict = self.getFeatures(game, player, ("Turn", 1))
        self.assertEquals(featsDict['GameStages.index(table.gameStage)'], 2)
        self.assertEquals(featsDict['h.myLastBet'], 40)
        self.assertEquals(featsDict['h.round_number'], 1)
        self.assertEquals(featsDict['len(table.other_active_players)'], 1)
        self.assertEquals(featsDict['len(table.other_players)'], 3)
        self.assertEquals(featsDict['table.currentBetValue'], 40)
        self.assertEquals(featsDict['table.currentCallValue'], 40)
        self.assertEquals(featsDict['table.first_caller'], 4)
        self.assertEquals(featsDict['table.first_raiser'], -1)
        self.assertEquals(featsDict['table.max_X'], 1)
        self.assertEquals(featsDict['table.myCallSum'], 20)
        self.assertEquals(featsDict['table.myFunds'], 30732)
        self.assertEquals(featsDict['table.myPos'], 4)
        self.assertEquals(featsDict['table.myRaiseSum'], 40)
        self.assertEquals(featsDict['table.nMyCalls'], 1)
        self.assertEquals(featsDict['table.nMyRaises'], 1)
        self.assertEquals(featsDict['table.round_pot_value'], 80)
        self.assertEquals(featsDict['table.second_caller'], 1)
        self.assertEquals(featsDict['table.second_raiser'], -1)
        self.assertEquals(featsDict['table.totalPotValue'], 140)
    
    def test3(self):
        with open("game-pickles/game-4.pkl", "rb") as encodeRec:
            game = load(encodeRec)
        player = game.players[0]
        featsDict = self.getFeatures(game, player, ("Flop", 1))
        self.assertEquals(featsDict['GameStages.index(table.gameStage)'], 1)
        self.assertEquals(featsDict['h.myLastBet'], 15)
        self.assertEquals(featsDict['h.round_number'], 1)
        self.assertEquals(featsDict['len(table.other_active_players)'], 1)
        self.assertEquals(featsDict['len(table.other_players)'], 2)
        self.assertEquals(featsDict['table.currentBetValue'], 15)
        self.assertEquals(featsDict['table.currentCallValue'], 15)
        self.assertEquals(featsDict['table.first_caller'], 3)
        self.assertEquals(featsDict['table.first_raiser'], 3)
        self.assertEquals(featsDict['table.max_X'], 1)
        self.assertEquals(featsDict['table.myCallSum'], 5)
        self.assertEquals(featsDict['table.myFunds'], 3900)
        self.assertEquals(featsDict['table.myPos'], 1)
        self.assertEquals(featsDict['table.myRaiseSum'], 35)
        self.assertEquals(featsDict['table.nMyCalls'], 1)
        self.assertEquals(featsDict['table.nMyRaises'], 3)
        self.assertEquals(featsDict['table.round_pot_value'], 60)
        self.assertEquals(featsDict['table.second_caller'], 1)
        self.assertEquals(featsDict['table.second_raiser'], 1)
        self.assertEquals(featsDict['table.totalPotValue'], 90)

    def test4(self):
        with open("game-pickles/game-4.pkl", "rb") as encodeRec:
            game = load(encodeRec)
        player = game.players[2]
        featsDict = self.getFeatures(game, player, ("Flop", 0))
        self.assertEquals(featsDict['GameStages.index(table.gameStage)'], 1)
        self.assertEquals(featsDict['h.myLastBet'], 15)
        self.assertEquals(featsDict['h.round_number'], 0)
        self.assertEquals(featsDict['len(table.other_active_players)'], 1)
        self.assertEquals(featsDict['len(table.other_players)'], 2)
        self.assertEquals(featsDict['table.currentBetValue'], 15)
        self.assertEquals(featsDict['table.currentCallValue'], 15)
        self.assertEquals(featsDict['table.first_caller'], 3)
        self.assertEquals(featsDict['table.first_raiser'], 3)
        self.assertEquals(featsDict['table.max_X'], 1)
        self.assertEquals(featsDict['table.myCallSum'], 10)
        self.assertEquals(featsDict['table.myFunds'], 1860)
        self.assertEquals(featsDict['table.myPos'], 3)
        self.assertEquals(featsDict['table.myRaiseSum'], 15)
        self.assertEquals(featsDict['table.nMyCalls'], 1)
        self.assertEquals(featsDict['table.nMyRaises'], 1)
        self.assertEquals(featsDict['table.round_pot_value'], 30)
        self.assertEquals(featsDict['table.second_caller'], 1)
        self.assertEquals(featsDict['table.second_raiser'], -1)
        self.assertEquals(featsDict['table.totalPotValue'], 60)
    
    def getFeatures(self, game, player, retAt ):
        h = History()
        table = self.dc.newTable(game, player, h)
        
        # Go through the game and save all data
        feats = None
        while type(feats) == type(None):
            feats = self.dc.playStage(game, player, table, h, retAt)
        return {FEATURES[i]:feats[i] for i in range(len(FEATURES))}
    
    
    def collectGameData(self):
        """This is how I collected the data for unittesting."""
        from pickle import dump
        import os
        ircParser = IrcHoldemDataParser("../../data")
        games = [ircParser.getRandomGame() for _ in list_of_numbers] 
        if not os.path.exists("game-pickles"):
            os.mkdir("game-pickles")
        for i in range(len(games)):
            with open("game-pickles/game-%d.pkl" % i, "wb") as outFile:
                dump(games[i], outFile)
        self.assertEqual(1, 1)
        
