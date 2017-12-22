
# Paths

DATA_DIR_PATH = "data"
IRC_DATA_PATH = "data/irc-data"
HAND_DATA_PATH = "data/hand-data"
GAME_VECTORS_PATH = "data/game-vectors"
POKER_BOT_DATA_PATH = "data/pb-data"
HOLDEM_PATHS_REGEX = r"holdem*"


# Features

FEATURES = [
    
    
    ]

# Poker info

# Actions
NA = "-"  # No action marker
QUIT = "Q"
BLIND = "B"  # 
ALL_IN = "A"  # 
CALL = "c"
CHECK = "k"
FOLD = "f"
BET = "b"
RAISE = "r"
KICKED= "K"

IRRELEVANT_FEATS = [
    
    
    
    ]

# Stages
GameStages = ['PreFlop', 'Flop', 'Turn', 'River', 'Showdown']
PreFlop, Flop, Turn, River, Showdown = GameStages


# Equity caches:

# These are not constants,
# because they are supposed to be updated
relEquityCache = None
globalEquityCache = None

defaultStrategy = {'_id': 0, 'turn_betting_condidion_1': 1, 'PreFlopMinCallEquity': 0.54, 'secondRoundAdjustmentPreFlop': 0.1, 'minBullyEquity': 1.0, 'range_utg3': 0.45, 'alwaysCallEquity': 0.98, 'range_utg4': 0.5, 'FlopCallPower': 1.0, 'bigBlind': 0.04, 'differentiate_reverse_sheet': 0, 'river_bluffing_condidion_1': 0, 'potAdjustment': 1.0, 'turn_bluffing_condidion_2': 1, 'flop_betting_condidion_1': 1, 'maxPotAdjustment': 0.02, 'range_utg5': 0.5, 'Strategy': 'nd_pp_1', 'BetPlusInc': 3.0, 'river_bluffing_condidion_2': 1, 'RiverCallPower': 1.0, 'turn_bluffing_condidion_1': 0, 'TurnCheckDeceptionMinEquity': 1.0, 'RiverBluffMinEquity': 1.0, 'FlopCheckDeceptionMinEquity': 0.85, 'RiverMinCallEquity': 0.85, 'initialFunds': 6.36, 'max_abs_fundchange': 5.0, 'always_call_low_stack_multiplier': 12.0, 'CoveredPlayersCallLikelihoodFlop': 0.1, 'use_pot_multiples': 0, 'maxPotAdjustmentPreFlop': 0.03, 'secondRiverBetPotMinEquity': 0.98, 'out_multiplier': 2.0, 'PreFlopMinBetEquity': 0.62, 'bullyDivider': 1.0, 'river_betting_condidion_1': 1, 'pokerSite': 'PP', 'FlopBluffMaxEquity': 0.7, 'considerLastGames': 100.0, 'pre_flop_equity_reduction_by_position': 0.01, 'RiverCheckDeceptionMinEquity': 1.0, 'FlopMinCallEquity': 0.58, 'use_relative_equity': 0, 'range_multiple_players': 0.25, 'computername': 'NICOLAS-ASUS', 'opponent_raised_without_initiative_river': 1, 'RiverBetPower': 1.0, 'maxBullyEquity': 0.0, 'FlopBluffMinEquity': 0.55, 'secondRoundAdjustmentPowerIncrease': 2.0, 'range_utg1': 0.4, 'RiverMinBetEquity': 0.82, 'range_utg2': 0.45, 'opponent_raised_without_initiative_flop': 1, 'PreFlopMaxBetEquity': 0.8, 'FlopBetPower': 1.0, 'TurnMinCallEquity': 0.68, 'betPotRiverEquity': 0.95, 'preflop_override': 0, 'flop_bluffing_condidion_1': 1, 'betPotRiverEquityMaxBBM': 30.0, 'secondRoundAdjustment': 0.05, 'minimum_bet_size': 3.0, 'TurnBetPower': 1.0, 'PreFlopCallPower': 1.0, 'pre_flop_equity_increase_if_bet': 0.05, 'gather_player_names': 0, 'collusion': 1, 'PreFlopBetPower': 1.0, 'strategyIterationGames': 1013.0, 'FlopMinBetEquity': 0.67, 'minimumLossForIteration':-100.0, 'potAdjustmentPreFlop': 1.0, 'range_utg0': 0.4, 'TurnBluffMinEquity': 1.0, 'TurnMinBetEquity': 0.68, 'pre_flop_equity_increase_if_call': 0.02, 'RiverBluffMaxEquity': 1.0, 'initialFunds2': 3.4, 'smallBlind': 0.02, 'TurnCallPower': 1.0, 'opponent_raised_without_initiative_turn': 1, 'TurnBluffMaxEquity': 1.0}

