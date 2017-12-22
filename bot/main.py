

import matplotlib
matplotlib.use('Qt5Agg')
# matplotlib.use('agg')
import pandas as pd
import time
import numpy as np
import pymongo
import os, sys
sys.path=["C:\\Users\\Omer ASUS\\git\\POD"]+sys.path
# os.environ['KERAS_BACKEND']='theano'
import logging.handlers
import pytesseract
import threading
import datetime
from PIL import Image
from PyQt5 import QtWidgets, QtGui
from configobj import ConfigObj
from bot.gui.gui_qt_ui import Ui_Pokerbot
from bot.gui.gui_qt_logic import UIActionAndSignals
from bot.tools.mongo_manager import StrategyHandler, UpdateChecker, GameLogger
from bot.table_analysers.table_screen_based import TableScreenBased
from bot.decisionmaker.current_hand_memory import History, CurrentHandPreflopState
from bot.decisionmaker.montecarlo_python import run_montecarlo_wrapper
from bot.decisionmaker.decisionmaker import Decision
from bot.tools.mouse_mover import MouseMoverTableBased


version = 3.04




class ThreadManager(threading.Thread):
    def __init__(self, threadID, name, counter, gui_signals):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.gui_signals = gui_signals
        self.logger = logging.getLogger('main') # Displays game info in gui
        self.logger.setLevel(logging.DEBUG)
        # Start a mongodb based game logger
        self.game_logger = GameLogger()

    def update_most_gui_items(self, preflop_state, p, m, table, d, h, gui_signals):
        try:
            sheet_name = table.preflop_sheet_name
        except:
            sheet_name = ''
        gui_signals.signal_decision.emit(str(d.decision + " " + sheet_name))
        gui_signals.signal_status.emit(d.decision)
        range2 = ''
        if hasattr(table, 'reverse_sheet_name'):
            range = table.reverse_sheet_name
            if hasattr(preflop_state, 'range_column_name'):
                range2 = " " + preflop_state.range_column_name + ""

        else:
            range = str(m.opponent_range)
        if range == '1': range = 'All cards'

        if table.gameStage != 'PreFlop' and p.selected_strategy['preflop_override']:
            sheet_name=preflop_state.preflop_sheet_name

        gui_signals.signal_label_number_update.emit('equity', str(np.round(table.abs_equity * 100, 2)) + "%")
        gui_signals.signal_label_number_update.emit('required_minbet', str(np.round(table.minBet,2)))
        gui_signals.signal_label_number_update.emit('required_mincall', str(np.round(table.minCall,2)))
        # gui_signals.signal_lcd_number_update.emit('potsize', table.totalPotValue)
        gui_signals.signal_label_number_update.emit('gamenumber',
                                                    str(int(self.game_logger.get_game_count(p.current_strategy))))
        gui_signals.signal_label_number_update.emit('assumed_players', str(int(table.assumedPlayers)))
        gui_signals.signal_label_number_update.emit('calllimit', str(np.round(d.finalCallLimit,2)))
        gui_signals.signal_label_number_update.emit('betlimit', str(np.round(d.finalBetLimit,2)))
        gui_signals.signal_label_number_update.emit('runs', str(int(m.runs)))
        gui_signals.signal_label_number_update.emit('sheetname', sheet_name)
        gui_signals.signal_label_number_update.emit('collusion_cards', str(m.collusion_cards))
        gui_signals.signal_label_number_update.emit('mycards', str(table.mycards))
        gui_signals.signal_label_number_update.emit('tablecards', str(table.cardsOnTable))
        gui_signals.signal_label_number_update.emit('opponent_range', str(range) + str(range2))
        gui_signals.signal_label_number_update.emit('mincallequity', str(np.round(table.minEquityCall, 2) * 100) + "%")
        gui_signals.signal_label_number_update.emit('minbetequity', str(np.round(table.minEquityBet, 2) * 100) + "%")
        gui_signals.signal_label_number_update.emit('outs', str(d.outs))
        gui_signals.signal_label_number_update.emit('initiative', str(table.other_player_has_initiative))
        gui_signals.signal_label_number_update.emit('round_pot', str(np.round(table.round_pot_value,2)))
        gui_signals.signal_label_number_update.emit('pot_multiple', str(np.round(d.pot_multiple,2)))

        if table.gameStage != 'PreFlop' and p.selected_strategy['use_relative_equity']:
            gui_signals.signal_label_number_update.emit('relative_equity', str(np.round(table.relative_equity,2) * 100) + "%")
            gui_signals.signal_label_number_update.emit('range_equity', str(np.round(table.range_equity,2) * 100) + "%")
        else:
            gui_signals.signal_label_number_update.emit('relative_equity', "")
            gui_signals.signal_label_number_update.emit('range_equity', "")



        # gui_signals.signal_lcd_number_update.emit('zero_ev', round(d.maxCallEV, 2))

        gui_signals.signal_pie_chart_update.emit(table.winnerCardTypeList)
        gui_signals.signal_curve_chart_update1.emit(h.histEquity, h.histMinCall, h.histMinBet, table.equity,
                                                    table.minCall, table.minBet,
                                                    'bo',
                                                    'ro')

        gui_signals.signal_curve_chart_update2.emit(table.power1, table.power2, table.minEquityCall, table.minEquityBet,
                                                    table.smallBlind, table.bigBlind,
                                                    table.maxValue_call,table.maxValue_bet,
                                                    table.maxEquityCall, table.max_X, table.maxEquityBet)

    def run(self):
        # Start a history from an excel file
        # File url defaults to "preflop.xlsx"
        hist = History()
        preflop_url, preflop_url_backup = updateChecker.get_preflop_sheet_url()
        try:
            hist.preflop_sheet = pd.read_excel(preflop_url, sheetname=None)
        except:
            hist.preflop_sheet = pd.read_excel(preflop_url_backup, sheetname=None)

        self.game_logger.clean_database()

        p = StrategyHandler()
        p.read_strategy() # Loads strategy dictionary
        preflop_state = CurrentHandPreflopState()

        while True:
            if self.gui_signals.pause_thread:
                while self.gui_signals.pause_thread == True:
                    time.sleep(1)
                    if self.gui_signals.exit_thread == True: sys.exit()
            
            # Try to initialize table
            ready = False
            while (not ready):
                p.read_strategy()
                table = TableScreenBased(p, gui_signals, self.game_logger, version)
                mouse = MouseMoverTableBased(p.selected_strategy['pokerSite'])
                mouse.move_mouse_away_from_buttons_jump

                ready = table.take_screenshot(True, p) and \
                        table.get_top_left_corner(p) and \
                        table.check_for_captcha(mouse) and \
                        table.get_lost_everything(hist, table, p, gui_signals) and \
                        table.check_for_imback(mouse) and \
                        table.get_my_cards(hist) and \
                        table.get_new_hand(mouse, hist, p) and \
                        table.get_table_cards(hist) and \
                        table.upload_collusion_wrapper(p, hist) and \
                        table.get_dealer_position() and \
                        table.get_snowie_advice(p, hist) and \
                        table.check_fast_fold(hist, p, mouse) and \
                        table.check_for_button() and \
                        table.get_round_number(hist) and \
                        table.init_get_other_players_info() and \
                        table.get_other_player_names(p) and \
                        table.get_other_player_funds(p) and \
                        table.get_other_player_pots() and \
                        table.get_total_pot_value(hist) and \
                        table.get_round_pot_value(hist) and \
                        table.check_for_checkbutton() and \
                        table.get_other_player_status(p, hist) and \
                        table.check_for_call() and \
                        table.check_for_betbutton() and \
                        table.check_for_allincall() and \
                        table.get_current_call_value(p) and \
                        table.get_current_bet_value(p)
            
            # While user wants to continue
            if not self.gui_signals.pause_thread:
                config = ConfigObj("config.ini")
                # Evaluate odds
                m = run_montecarlo_wrapper(p, self.gui_signals, config, ui, table, self.game_logger, preflop_state, hist)
                
                # Decide what to do
                # @TODO: make it more sophisticated by taking more information
                # into account, e.g. estimates of other players' equity
                decision = Decision(table, hist, p, self.game_logger)
                decision.make_decision(table, hist, p, self.logger, self.game_logger)
                
                
                if self.gui_signals.exit_thread: sys.exit()
                
                
                # Update game state
                # @TODO: add more game info in game states
                self.update_most_gui_items(preflop_state, p, m, table, decision, hist, self.gui_signals)
                
                
                # Report game state
                self.logger.info(
                    "Equity: " + str(table.equity * 100) + "% -> " + str(int(table.assumedPlayers)) + " (" + str(
                        int(table.other_active_players)) + "-" + str(int(table.playersAhead)) + "+1) Plr")
                self.logger.info("Final Call Limit: " + str(decision.finalCallLimit) + " --> " + str(table.minCall))
                self.logger.info("Final Bet Limit: " + str(decision.finalBetLimit) + " --> " + str(table.minBet))
                self.logger.info(
                    "Pot size: " + str((table.totalPotValue)) + " -> Zero EV Call: " + str(round(decision.maxCallEV, 2)))
                self.logger.info("+++++++++++++++++++++++ Decision: " + str(decision.decision) + "+++++++++++++++++++++++")

                # Perform action
                mouse_target = decision.decision
                if mouse_target == 'Call' and table.allInCallButton:
                    mouse_target = 'Call2'
                mouse.mouse_action(mouse_target, table.tlc)

                table.time_action_completed = datetime.datetime.utcnow()
                
                filename = str(hist.GameID) + "_" + str(table.gameStage) + "_" + str(hist.round_number) + ".png"
                self.logger.debug("Saving screenshot: " + filename)
                pil_image = table.crop_image(table.entireScreenPIL, table.tlc[0], table.tlc[1], table.tlc[0] + 950, table.tlc[1] + 650)
                pil_image.save("log/screenshots/" + filename)

                self.gui_signals.signal_status.emit("Logging data")
                
                # Log game into mongodb
                t_log_db = threading.Thread(name='t_log_db', target=self.game_logger.write_log_file, args=[p, hist, table, decision])
                t_log_db.daemon = True
                t_log_db.start()
                # self.game_logger.write_log_file(p, hist, table, decision)

                # Update game history 
                hist.previousPot = table.totalPotValue
                hist.histGameStage = table.gameStage
                hist.histDecision = decision.decision
                hist.histEquity = table.equity
                hist.histMinCall = table.minCall
                hist.histMinBet = table.minBet
                hist.hist_other_players = table.other_players
                hist.first_raiser = table.first_raiser
                hist.first_caller = table.first_caller
                hist.previous_decision = decision.decision
                hist.lastRoundGameID = hist.GameID
                hist.previous_round_pot_value=table.round_pot_value
                hist.last_round_bluff = False if table.currentBluff == 0 else True
                if table.gameStage == 'PreFlop':
                    preflop_state.update_values(table, decision.decision, hist, decision)
                self.logger.info("=========== round end ===========")
                

# ==== MAIN PROGRAM =====

def run_poker():
    fh = logging.handlers.RotatingFileHandler('log/pokerprogram.log', maxBytes=1000000, backupCount=10)
    fh.setLevel(logging.DEBUG)
    fh2 = logging.handlers.RotatingFileHandler('log/pokerprogram_info_only.log', maxBytes=1000000, backupCount=5)
    fh2.setLevel(logging.INFO)
    er = logging.handlers.RotatingFileHandler('log/errors.log', maxBytes=2000000, backupCount=2)
    er.setLevel(logging.WARNING)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARNING)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    fh2.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    er.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    ch.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))

    root = logging.getLogger()
    root.addHandler(fh)
    root.addHandler(fh2)
    root.addHandler(ch)
    root.addHandler(er)

    print(
        "This is a testversion and error messages will appear here. The user interface has opened in a separate window.")
    # Back up the reference to the exceptionhook
    sys._excepthook = sys.excepthook
    global updateChecker
    updateChecker = UpdateChecker()
    updateChecker.check_update(version)
    def exception_hook(exctype, value, traceback):
        # Print the error and traceback
        logger = logging.getLogger('main')
        logger.setLevel(logging.DEBUG)
        print(exctype, value, traceback)
        logger.error(str(exctype))
        logger.error(str(value))
        logger.error(str(traceback))
        # Call the normal Exception hook after
        sys.__excepthook__(exctype, value, traceback)
        sys.exit(1)
    # Set the exception hook to our wrapping function
    sys.__excepthook__ = exception_hook

    # check for tesseract
    try:
        pytesseract.image_to_string(Image.open('pics/PP/call.png'))
    except Exception as e:
        print(e)
        print(
            "Tesseract not installed. Please install it into the same folder as the pokerbot or alternatively set the path variable.")
        # subprocess.call(["start", 'tesseract-installer/tesseract-ocr-setup-3.05.00dev.exe'], shell=True)
        sys.exit()

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()

    global ui
    ui= Ui_Pokerbot()
    ui.setupUi(MainWindow)
    MainWindow.setWindowIcon(QtGui.QIcon('icon.ico'))

    global gui_signals
    gui_signals = UIActionAndSignals(ui)

    t1 = ThreadManager(1, "Thread-1", 1, gui_signals)
    t1.start()
    MainWindow.show()
    try:
        sys.exit(app.exec_())
    except:
        print("Preparing to exit...")
        gui_signals.exit_thread = True


    pass

if __name__ == '__main__':
    run_poker()