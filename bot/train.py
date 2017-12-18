import matplotlib
import pandas as pd
import time
import numpy as np
from parse_irc_data import IrcHoldemDataParser
from bot.table_analysers.base import Table
from bot.decisionmaker.montecarlo_python import MonteCarlo
from copy import copy
from bot.tools.mongo_manager import GameLogger, StrategyHandler
from bot.decisionmaker.current_hand_memory import History, \
    CurrentHandPreflopState
from bot.decisionmaker.decisionmaker import Decision
from bot.table_analysers.table_screen_based import TableScreenBased
from random import choice

import os, sys
sys.path = ["C:\\Users\\Omer ASUS\\git\\POD\\"] + sys.path
# os.environ['KERAS_BACKEND']='theano'

import logging.handlers
import pytesseract
import threading
import datetime
from PIL import Image, ImageGrab
from PyQt5 import QtWidgets, QtGui
from configobj import ConfigObj


version = 3.04

DATA_DIR_PATH = "../data"

GAME_STAGES = ["PreFlop", "Flop", "Turn", "River", "Showdown"]

ircParser = IrcHoldemDataParser(DATA_DIR_PATH)


def run_montecarlo_wrapper(p, config, t, L, preflop_state, h):
    # Prepare for montecarlo simulation to evaluate equity (probability of winning with given cards)
    m = MonteCarlo()

    logger = logging.getLogger('montecarlo')
    logger.setLevel(logging.DEBUG)

    if t.gameStage == "PreFlop":
        t.assumedPlayers = 2
        opponent_range = 1

    elif t.gameStage == "Flop":

        if t.isHeadsUp:
            for i in range(5):
                if t.other_players[i]['status'] == 1:
                    break
            n = t.other_players[i]['utg_position']
            logger.info("Opponent utg position: " + str(n))
            opponent_range = float(p.selected_strategy['range_utg' + str(n)])
        else:
            opponent_range = float(p.selected_strategy['range_multiple_players'])

        t.assumedPlayers = t.other_active_players - int(round(t.playersAhead * (1 - opponent_range))) + 1

    else:

        if t.isHeadsUp:
            for i in range(5):
                if t.other_players[i]['status'] == 1:
                    break
            n = t.other_players[i]['utg_position']
            logger.info("Opponent utg position: " + str(n))
            opponent_range = float(p.selected_strategy['range_utg' + str(n)])
        else:
            opponent_range = float(p.selected_strategy['range_multiple_players'])

        t.assumedPlayers = t.other_active_players + 1

    t.assumedPlayers = min(max(t.assumedPlayers, 2), 4)

    t.PlayerCardList = []
    t.PlayerCardList.append(t.mycards)
    t.PlayerCardList_and_others = copy(t.PlayerCardList)

    ghost_cards = ''
    m.collusion_cards = ''

    if p.selected_strategy['collusion'] == 1:
        collusion_cards, collusion_player_dropped_out = L.get_collusion_cards(h.game_number_on_screen, t.gameStage)

        if collusion_cards != '':
            m.collusion_cards = collusion_cards
            if not collusion_player_dropped_out:
                t.PlayerCardList_and_others.append(collusion_cards)
                logger.info("Collusion found, player still in game. " + str(collusion_cards))
            elif collusion_player_dropped_out:
                logger.info("COllusion found, but player dropped out." + str(collusion_cards))
                ghost_cards = collusion_cards
        else:
            logger.debug("No collusion found")

    else:
        m.collusion_cards = ''

    if t.gameStage == "PreFlop":
        maxRuns = 1000
    else:
        maxRuns = 7500

    if t.gameStage != 'PreFlop':
        try:
            for abs_pos in range(5):
                if t.other_players[abs_pos]['status'] == 1:
                    sheet_name = preflop_state.get_reverse_sheetname(abs_pos, t, h)
                    ranges = preflop_state.get_rangecards_from_sheetname(abs_pos, sheet_name, t, h, p)
                    # logger.debug("Ranges from reverse table: "+str(ranges))

                    # the last player's range will be relevant
                    if t.isHeadsUp == True:
                        opponent_range = ranges

        except Exception as e:
            logger.error("Opponent reverse table failed: " + str(e))

    logger.debug("Running Monte Carlo")
    t.montecarlo_timeout = float(config['montecarlo_timeout'])
    timeout = t.mt_tm + t.montecarlo_timeout
    logger.debug("Used opponent range for montecarlo: " + str(opponent_range))
    logger.debug("maxRuns: " + str(maxRuns))
    logger.debug("Player amount: " + str(t.assumedPlayers))

    # calculate range equity
    if t.gameStage != 'PreFlop' and p.selected_strategy['use_relative_equity']:
        if p.selected_strategy['preflop_override'] and preflop_state.preflop_bot_ranges != None:
            t.player_card_range_list_and_others = t.PlayerCardList_and_others[:]
            t.player_card_range_list_and_others[0] = preflop_state.preflop_bot_ranges

            t.range_equity, _ = m.run_montecarlo(logger, t.player_card_range_list_and_others, t.cardsOnTable,
                                              int(t.assumedPlayers), maxRuns=maxRuns,
                                              ghost_cards=ghost_cards, timeout=timeout, opponent_range=opponent_range)
            t.range_equity = np.round(t.range_equity, 2)
            logger.debug("Range montecarlo completed successfully with runs: " + str(m.runs))
            logger.debug("Range equity (range for bot): " + str(t.range_equity))

    if preflop_state.preflop_bot_ranges == None and p.selected_strategy['preflop_override'] and t.gameStage != 'PreFlop':
        logger.error("No preflop range for bot, assuming 50% relative equity")
        t.range_equity = .5


    # run montecarlo for absolute equity
    t.abs_equity, _ = m.run_montecarlo_without_ui(logger, t.PlayerCardList_and_others, t.cardsOnTable, int(t.assumedPlayers), maxRuns=maxRuns,
                     ghost_cards=ghost_cards, timeout=timeout, opponent_range=opponent_range)
    logger.debug("Cards Monte Carlo completed successfully with runs: " + str(m.runs))
    logger.info("Absolute equity (no ranges for bot) " + str(np.round(t.abs_equity, 2)))

    if t.gameStage == "PreFlop":
        crd1, crd2 = m.get_two_short_notation(t.mycards)
        if crd1 in m.preflop_equities:
            m.equity = m.preflop_equities[crd1]
        elif crd2 in m.preflop_equities:
            m.equity = m.preflop_equities[crd2]
        elif crd1 + 'O' in m.preflop_equities:
            m.equity = m.preflop_equities[crd1 + 'O']
        else:
            logger.warning("Preflop equity not found in lookup table: " + str(crd1))
        t.abs_equity = m.equity

    t.abs_equity = np.round(t.abs_equity, 2)
    t.winnerCardTypeList = m.winnerCardTypeList

    m.opponent_range = opponent_range

    if t.gameStage != 'PreFlop' and p.selected_strategy['use_relative_equity']:
        t.relative_equity = np.round(t.abs_equity / t.range_equity / 2, 2)
        logger.info("Relative equity (equity/range equity/2): " + str(t.relative_equity))
    else:
        t.range_equity = ''
        t.relative_equity = ''
    return m


from unittest import TestCase
import numpy as np

class IrcTrainer:

    def nextTable(h, p, game_logger, version):
        game = ircParser.getRandomGame()
        gameStage = "PreFlop"
        player = choice(game.players) # The player we're imitating
        # Init history
        h.round_number = 0
        preflop_url = "decisionmaker/preflop.xlsx"
        h.preflop_sheet_name = "preflop.xlsx"
        h.preflop_sheet = pd.read_excel(preflop_url, sheetname=None)
        h.myLastBet = 0
        # Initialize table from game state
        
    
        # (Right now this is dummy, just to show which variables should be initialized)
        table = TableScreenBased(p, None, game_logger, version)
        table.entireScreenPIL=ImageGrab.grab()
        table.gameStage = gameStage
        table.first_raiser = ""
        table.second_raiser = ""
        table.first_caller = ""
        table.second_caller = ""
        table.mycards = ['QD', 'QS']  # (a) translate irc cards to bot cards, and (b) what to do if player has no cards?
        table.other_players = game.players
        table.round_pot_value = 0
        table.currentCallValue = 0
        table.checkButton = 0
        table.currentBetValue = 0
        table.relative_equity = 0
        table.position_utg_plus = 0
        table.abs_equity = 0
        table.cardsOnTable = game.boardCards # change to correct format
        table.first_raiser_utg = 0
        table.first_caller_utg = 0
        table.second_raiser_utg = 0
    #     table.minCallAmountIfAboveLimit=0
    #     table.minEquityCall=0
    #     table.maxEquityCall=0
        table.totalPotValue = 0
        table.max_X = 0
        table.other_player_has_initiative = 0
        table.potStretch =0
        table.isHeadsUp = 0
        table.PlayerNames = 0
        table.myFunds = player.bankroll
        table.allInCallButton = True
        table.checkButton = True
        table.betButton = True
        
        return table

    def train():
        config = ConfigObj("config.ini")
        game_logger = GameLogger()
        logger = logging.getLogger('main')
        h = History()
        
        
        p = StrategyHandler()
        p.read_strategy()
        preflop_state = CurrentHandPreflopState()
        
        t = makeTable(h, p, game_logger, version)
        
        
        d = Decision(t, h, p, game_logger)
        d.make_decision(t, h, p, logger, game_logger)
        print(d)
        h.previousPot = t.totalPotValue
        h.histGameStage = t.gameStage
        h.histDecision = d.decision
        h.histEquity = t.equity
        h.histMinCall = t.minCall
        h.histMinBet = t.minBet
        h.hist_other_players = t.other_players
        h.first_raiser = t.first_raiser
        h.first_caller = t.first_caller
        h.previous_decision = d.decision
        h.lastRoundGameID = h.GameID
        h.previous_round_pot_value = t.round_pot_value
        h.last_round_bluff = False if t.currentBluff == 0 else True
        if t.gameStage == 'PreFlop':
            preflop_state.update_values(t, d.decision, h, d)
        
        

if __name__ == '__main__':
    train()
