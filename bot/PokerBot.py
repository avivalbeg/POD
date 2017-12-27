import os, shutil
import time, threading
import ntpath
from os.path import splitext, join, exists
from PIL import ImageGrab
import sys
from pygame.locals import *
from PIL import Image
import cv2
import numpy as np
import datetime
import inspect
from pprint import pprint
from pandas.core.frame import DataFrame
import pandas
from collections import OrderedDict
from bot.tools.mouse_mover import MouseMoverTableBased
from util import DummyLogger, getRank, getHandType, getEquity, emptyDir
import eval7
from pytesseract.pytesseract import image_to_string
from constants import FOLD, CALL
import re

startTime = time.time()


SCREENSHOTS_DIR_PATH = "screenshots"
PICS_DIR_PATH = "pics"
COORDS_PATH = "coordinates.txt"
POKER_SITES = ["PP", "PS"]

TLC = "topleft"
IMBACK = "imback"
CHECK_BUTTON = "check"
CALL_BUTTON = "call"
DOLLAR_SIGN = "smalldollarsign1"
ALL_IN_BUTTON = "allincallbutton"
LOST_EVERYTHING = "lostEverything"
DEALER = "dealer"
BET_BUTTON = "betbutton" 


SCREEN_ITEMS = [
        TLC,
        IMBACK,
        CHECK_BUTTON,
        CALL_BUTTON,
        DOLLAR_SIGN,
        ALL_IN_BUTTON,
        LOST_EVERYTHING,
        DEALER,
        BET_BUTTON        
                ] 

DENOMS = "23456789TJQKA"
SUITES = "CDHS"
CARDS = [d + s for d in DENOMS for s in SUITES]

GAME_OBJECTS = [
    'gameStage',
    'roundNumber',
    'nPlayers',
    'nActivePlayers',
    'absEquity',
    'absRank',
    'absHandType',
    'firstRaiser',
    'secondRaiser',
    'firstCaller',
    'secondCaller',
    'lastRaiser',
    'totalPotValue',
    'stagePotValue',
    'lastBetValue',
    'currentCallValue',
    'tableCards',
    'otherPlayers',
    
    # Bot objects
    'myPos',
    'myCards',
    'myFunds',
    'myLastBet',
    'myLastCall',
    'myLastAction',
    'myRaiseSum',
    'myCallSum',
    'myNRaises',
    'myNCalls',
    'myNChecks',
    'myEquity',
    'myRank',
    'myHandType',
    ]

def inputThread(l):
    derp = input("Type 'q' to stop at any time.\n")
    if derp == "q" and l:
        l.pop()

class PokerBot:
    def __init__(self, pokerSite, saveScreenshots=False):
        emptyDir(SCREENSHOTS_DIR_PATH)
        self._saveScreenshots = saveScreenshots
        self._curScreen = None
        self._pokerSite = pokerSite
        self.loadCoordinates()
        self._gameState = DataFrame({obj:[] for obj in GAME_OBJECTS})
        
    def start(self, debugImage=None):
        
        print("Poker bot running...")
        
        # Listen to user
        run = [1]  # This pops when q is pressed, leading to program stopping
        if not debugImage:
            exit_t = threading.Thread(name="exit_t", target=inputThread, args=(run,))
            exit_t.daemon = True
            exit_t.start()
        
        self._mouseMover = MouseMoverTableBased(self._pokerSite)
        
        while run:
            time.sleep(.1)
        
            # save screenshots periodically
            self.saveScreenshots()
            
            # Look for items on screen until user wants to exit or 
            # items found
            ready = False
            while run and not ready:
                # Take screenshot
                if debugImage:
                    self._curScreen = Image.open(debugImage)
                else:
                    self._curScreen = ImageGrab.grab()
                    
                # Try to initialize table objects
                ready = self.updateGameStateFromScreen()
                
                # If debugging an image, we want to quit after one iteration
                if debugImage:
                    run.pop()
            # Do stuff
            if ready:
                decision = self.makeDecision()
                print("Decision: " + decision)
                self.moveMouse(decision)
                self.log()
                
    def log(self):
        pass
    
    def updateGameStateFromScreen(self):
        """Look for all items and cards on the
        screen and set the bot's attributes accordingly."""
        
        
        # Init screen items
        for item in SCREEN_ITEMS:
            points = self.findItemOnScreen(item)
            setattr(self, item, points)
        if not getattr(self, TLC):
            return False
        # Init game objects
        # Get objects
        myCards = self.getMyCards()
        tableCards = self.getTableCards()
        currentCallValue = self.getCurrentCallValue()
        
        # Update game state data frame
        
        if [myCards] != list(self._gameState["myCards"]):
            self._gameState["myCards"] = [myCards]
            if myCards:
                print("New cards detected: " + " ".join(myCards))
                with open("cardsLog.txt", "a") as f:
                    print(" ".join(self._gameState["myCards"][0]), file=f)
        
        if [tableCards] != list(self._gameState["tableCards"]):
            print("Table cards: " + " ".join(tableCards))
            self._gameState["tableCards"] = [tableCards]
            if len(tableCards) == 0:
                self._gameState["gameStage"] = [0]
            if len(tableCards) == 3:
                self._gameState["gameStage"] = [1]
            if len(tableCards) == 4:
                self._gameState["gameStage"] = [2]
            if len(tableCards) == 5:
                self._gameState["gameStage"] = [3]
            print("Game stage: " + str(self._gameState["gameStage"][0]))

        if [currentCallValue] != list(self._gameState["currentCallValue"]):
            currentCallValue < float("inf") and print("Call value: " + str(currentCallValue))
            self._gameState["currentCallValue"] = [currentCallValue]
            
        self._gameState["myEquity"] = getEquity(self._gameState["myCards"][0] + self._gameState["tableCards"][0])
        self._gameState["absEquity"] = getEquity(self._gameState["tableCards"][0])

        self._gameState["myHandType"] = getHandType(self._gameState["myCards"][0] + self._gameState["tableCards"][0])
        self._gameState["absHandType"] = getHandType(self._gameState["tableCards"][0])
        
        # Return True iff top left corner found
        if getattr(self, TLC) \
        and (getattr(self, CHECK_BUTTON)\
        or getattr(self, CALL_BUTTON)):
            return True
        else:
            return False
        
    # Decision making

    def makeDecision(self):
        """Make a decision about next move."""
        
        
        # This is just a dummy strategy meant to keep the bot 
        # alive for a while
        if getattr(self, CHECK_BUTTON):
            return "Check"
        elif getattr(self, CHECK_BUTTON):
            return "Imback"
        elif getattr(self, CALL_BUTTON)\
        and ((self._gameState["currentCallValue"][0] <= 300 \
              or (self._gameState["currentCallValue"][0] <= 800\
                  and self._gameState["gameStage"][0] == 3)\
              or (self._gameState["currentCallValue"][0] <= 400\
                  and self._gameState["gameStage"][0] == 2))
        and self._gameState["myHandType"][0] in ("Royal Flush", "High Flush", "Four of a Kind") \
        and not self._gameState["absHandType"][0] in ("Royal Flush", "High Flush", "Four of a Kind"))\
        or (self._gameState["currentCallValue"][0] <= 100\
            and ((self._gameState["gameStage"][0] == 0 \
                  and re.findall("[KQJA]", str(self._gameState["myCards"][0])))\
                 or (self._gameState["myHandType"][0] != "High Card"\
                 and self._gameState["myHandType"][0] == "High Card"))):
            return "Call"
        else:
            return "Fold"
    
    # Game representation
    
    def getGameState(self):
        return self._gameState
    
    # Mouse control
    
    def moveMouse(self, decision):
        self._mouseMover.move_mouse_away_from_buttons_jump()
        self._mouseMover.mouse_action(decision, getattr(self, TLC))
        time.sleep(2)
        pass
    # Screen parsing
    
    
    def getMyCards(self):
        """Returns a list of cards the bot has right now."""
        img = self.imageFromField("get_my_cards")
        # Look for each card
        cards = []
        for card in CARDS:
            target = Image.open(join(PICS_DIR_PATH, self._pokerSite, card + ".png"))
            template = cv2.cvtColor(np.array(target), cv2.COLOR_BGR2RGB)
            method = eval('cv2.TM_SQDIFF_NORMED')

            res = cv2.matchTemplate(img, template, method)

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if min_val < 0.01:
                cards.append(card)
        
        return cards
    
    def getTableCards(self):
        img = self.imageFromField("get_table_cards")
        
        # Look for each card
        cardsOnTable = []
        for card in CARDS:
            target = Image.open(join(PICS_DIR_PATH, self._pokerSite, card + ".png"))
            template = cv2.cvtColor(np.array(target), cv2.COLOR_BGR2RGB)
            method = eval('cv2.TM_SQDIFF_NORMED')
            res = cv2.matchTemplate(img, template, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if min_val < 0.01:
                cardsOnTable.append(card)
        return cardsOnTable
    
    def getCurrentCallValue(self):
        func_dict = self.coo["get_current_call_value"][self._pokerSite]
        pil_image = self.crop_image(self._curScreen,
                                    getattr(self, TLC)[0] + func_dict['x1'],
                                    getattr(self, TLC)[1] + func_dict['y1'],
                                    getattr(self, TLC)[0] + func_dict['x2'],
                                    getattr(self, TLC)[1] + func_dict['y2'])
        
        if not getattr(self, CHECK_BUTTON):
            callValue = image_to_string(pil_image)
        elif getattr(self, CHECK_BUTTON):
            callValue = 0

        if type(callValue) == type(''):
            try:
                callValue = eval(callValue)
            except:
                # Failed to identify call value
                # To err on the safe side, we assume it's 
                # high, not low
                callValue = float("inf")
        
        return callValue
    
    # Image processing
    
    def imageFromField(self, field):
        """Get coordinate field name, return a crop of the current
        image from its position on the screen."""
        
        func_dict = self.coo[field][self._pokerSite]
        pil_image = self.crop_image(self._curScreen, getattr(self, TLC)[0] + func_dict['x1'], getattr(self, TLC)[1] + func_dict['y1'],
                                        getattr(self, TLC)[0] + func_dict['x2'], getattr(self, TLC)[1] + func_dict['y2'])
        pil_image.save(join("savedScreenshots", field + ".png"))
        
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_BGR2RGB)
    
    def findItemOnScreen(self, item):
        """Given an item name (e.g. callButton),
        look for it on the screen and return coordinates if it was found, 
        None otherwise."""
        def helper(item):    
            # Load item template from image and look for it on the table
            img = cv2.cvtColor(np.array(self._curScreen), cv2.COLOR_BGR2RGB)
            target = Image.open(join(PICS_DIR_PATH, self._pokerSite, item + ".png"))
            template = cv2.cvtColor(np.array(target), cv2.COLOR_BGR2RGB)
            count, points, bestfit, minimum_value = self.findTemplateOnScreen(template, img, 0.01)
            
            if count == 1:
                return points[0]
            else:
                return None
        points = helper(item)
        # Try other pics with similar names
        if not points:
            for i in range(1, 10):
                if exists(join(PICS_DIR_PATH, self._pokerSite, item + str(i) + ".png")):
                    points = helper(item + str(i))
                    if points:
                        break
        
        return points
    
    def findTemplateOnScreen(self, template, screenshot, threshold):
        """Given a cv2 template, find its coordinates on the screen."""
        
        method = eval('cv2.TM_SQDIFF_NORMED')
        
        # Apply template Matching
        res = cv2.matchTemplate(screenshot, template, method)
        loc = np.where(res <= threshold)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            bestFit = min_loc
        else:
            bestFit = max_loc

        count = 0
        points = []
        for pt in zip(*loc[::-1]):
            count += 1
            points.append(pt)
        return count, points, bestFit, min_val

    
    def crop_image(self, original, left, top, right, bottom):
        width, height = original.size  # Get dimensions
        cropped_example = original.crop((left, top, right, bottom))
        return cropped_example
        
    def saveScreenshots(self):
        """If bot runs on save screenshots mode, 
        save screenshots every 7 seconds and get rid of
        screenshots older than 200 seconds to avoid explosion."""
        if self._curScreen and self._saveScreenshots:
            now = int(time.time() - startTime)
            if now % 7 < 2:
                self.curScreen.save(join(SCREENSHOTS_DIR_PATH, "%d.png" % now))
                for f in os.listdir(SCREENSHOTS_DIR_PATH):
                    if (now - eval(splitext(ntpath.basename(f))[0])) > 200:
                        os.remove(join(SCREENSHOTS_DIR_PATH, f))
    
    
    
    def loadCoordinates(self):
        with open(COORDS_PATH, 'r') as inf:
            c = eval(inf.read())
            self.coo = c['screen_scraping']
    
    
def main():
    bot = PokerBot("PS")
    bot.start()
    
if __name__ == '__main__':
    main()
    quit()
