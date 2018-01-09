# -*- coding: utf-8 -*-
import re
import sys
import numpy as np
from collections import defaultdict

from matplotlib import pyplot
from sklearn.cluster.k_means_ import KMeans
from sklearn.cross_validation import train_test_split

from constants import *
from pandas.util.testing import isiterable
from pprint import pprint
import ntpath
from os.path import splitext
import eval7
from PIL import Image
from pytesseract.pytesseract import image_to_string
import os


class DummyLogger:
    def info(self, *args):
        pass


def oneHotEncCat(cat, cats):
    """One-hot encode a category from a list of possible categories."""
    ind = cats.index(cat)
    vec = np.zeros(len(cats))
    vec[ind] = 1
    return vec


def pad(matrix, n, nDims):
    """Pad a 2D matrix with NaN vectors up to certain size."""
    for _ in range(max([0, n - len(matrix)])):
        emptyRound = np.empty(nDims)
        emptyRound[:] = NAN_NUM  # Fill it with nothing
        matrix = np.vstack((matrix, emptyRound))
    return matrix


def emptyDir(path):
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def eval7CardFormat(card):
    d, s = card
    return d.upper() + s.lower()


def getRank(hand):
    hand = [eval7.Card(eval7CardFormat(c)) for c in hand]
    return eval7.evaluate(hand)


def getHandType(hand):
    return eval7.hand_type(getRank(hand))


def encodeRec(x):
    """
    Recursively encode the contents of a dictionary. 
    """

    if type(x) == type({}):
        return {k: encodeRec(v) for k, v in x.items()}
    elif type(x) == type(""):
        return x.encode(sys.stdout.encoding, errors='replace')
    else:
        return x


def splitIrcLine(line):
    return re.split("\s+", line.strip())


def tableRowTitle(obj, spacesBeforeSep=15, spacesAfterSep=10, sep="|"):
    """
    Creates a title-formed string from an object. For example:
    
    >> tableRowTitle("Flop") + "2/20"
    Flop           |          2/20
    """
    if len(str(obj)) > spacesBeforeSep:
        return str(obj)[:spacesBeforeSep - 4] + "... " + sep + (" " * spacesAfterSep)
    return str(obj) + " " * (spacesBeforeSep - len(str(obj))) + sep + (" " * spacesAfterSep)


def tableRowFromList(lst, cellSize=10, sep="|"):
    """Create a table row from list, with each cell centered and 
    of size cellSize, separated by sep"""

    cells = [str(x) if len(str(x)) <= cellSize else str(x)[:cellSize - 4] + "... " for x in lst]
    out = ""
    for cell in cells:
        if len(cell) % 2 != 0:
            cell += " "
        spaces = " " * int((cellSize - len(cell)) / 2)
        out += spaces + cell \
               + spaces + sep
    return out


def readVocab(path):
    """Reads a word list off a file where each line 
    is a word"""
    vocab = set()
    with open(path) as f:
        for line in f:
            vocab.add(line.strip())
    return vocab


def writeVocab(vocab, path):
    """Writes a set into a file line by line."""
    with open(path, "w") as f:
        for x in vocab:
            print(x, file=f)


def divOr0(x, y):
    if y != 0:
        return float(x) / float(y)
    else:
        return 0.


def getOrDefault(obj, item, dflt):
    try:
        return obj[item]
    except (KeyError, IndexError):
        return dflt


def ircCardToBotCard(card):
    return card[0] + card[1].upper()


def findCards(string):
    return re.findall("-?[a-zA-Z\d][a-zA-Z\d]", str(string))


def isNumStr(x):
    return type(x) == type("") and re.match("-?\d*((\.\d*)|\d+)", x)


def makeFeatureVector(dic, n=0, whitelist=[], blacklist=[], strDic={}, normFeatures=[], norm=1.):
    """Turn an object into a feature vector."""

    if not n:
        n = len(dic)
    vector = np.zeros(n)
    pprint(dic.keys())
    if norm == 0:
        raise ValueError("Cannot normalize with respect to 0.")
    for i, (feature, value) in enumerate(sorted(dic.items())):

        if feature in blacklist:
            continue

        value = featurize(value)

        if feature in normFeatures:
            value /= norm
            assert value <= 1

        if isiterable(value):
            raise ValueError("Cannot use iterable as feature value")

        vector[i] = value
    return vector


def featurize(value, strDic={}):
    """Turn some value into a numerical feature."""

    if isNumStr(value):
        return eval(value)

    if type(value) == type(""):
        if value in ("nan", "NaN", "None"):
            return NAN_NUM
        elif value in ("True", "False"):
            return int(eval(value))
        elif not value:
            return 0
        else:
            return strDic[value]

    if type(value) in (type(np.nan), type(None)):
        return NAN_NUM

    if type(value) == type(False):
        return int(value)

    return value


def baseName(filePath):
    """Get the base name of a file path."""
    return splitext(ntpath.basename(filePath))[0]


def cleanMdbDic(dic):
    cleaned = {}
    for field, val in dic.items():

        if type(val) == type(b""):
            val = val.decode(sys.stdout.encoding)

        if val == "False": val = 0
        if val == "True": val = 1
        if val == "nan" or not val: val = NAN_NUM
        if val in GameStages: val = GameStages.index(val)

        if field == "other_players": val = str(val)
        if field == "PlayerCardList": val = " ".join(eval(val))
        if field == "PlayerCardList_and_others": val = " ".join(re.findall("[a-zA-Z\d]+", val))
        cleaned[field] = [val]

    return cleaned


def joinLists(lists):
    bigList = []
    for l in lists:
        bigList.extend(l)
    return bigList


padSequence = lambda mat, n: np.pad(mat[-n:], ((0, max([0, n - len(mat)])), (0, 0)), 'constant')


def makeFile(path):
    open(path, "w").close()
    return open(path, "ab")


def divXy(Xy, axis):
    X, y = np.split(Xy, (-1,), axis)
    y = np.max(np.squeeze(y, axis), axis - 1)
    return X, y


def trainDevTestPrep(array):

    # Split into train,dev,test
    train_dev, test = train_test_split(array)
    train, dev = train_test_split(train_dev)

    # Split to input and target
    trainX, train_y = divXy(train, 2)
    devX, dev_y = divXy(dev, 2)
    testX, test_y = divXy(test, 2)

    return trainX,train_y,devX,dev_y,testX,test_y
