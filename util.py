# -*- coding: utf-8 -*-
import re
import sys

def encodeRec(x):
    """
    Recursively encode the contents of a dictionary. 
    """
    
    if type(x)==type({}):
        return {k:encodeRec(v) for k,v in x.items()}
    elif type(x)==type(""):
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
        return str(obj)[:spacesBeforeSep - 4] + "... " + sep + (" " *spacesAfterSep)
    return str(obj) + " "*(spacesBeforeSep - len(str(obj))) + sep + (" " *spacesAfterSep)

def tableRowFromList(lst, cellSize=10, sep="|"):
    """Create a table row from list, with each cell centered and 
    of size cellSize, separated by sep""" 

    cells = [x if len(x) <= cellSize else x[:cellSize - 4] + "... " for x in lst ]
    out = ""
    for cell in cells:
        if len(cell) % 2 != 0:
            cell += " "
        spaces = " "*int((cellSize - len(cell)) / 2)
        out += spaces + cell \
        + spaces + sep
    return out



def divOr0(x,y):
    if y!=0:
        return float(x)/float(y)
    else:
        return 0.

def getOrDefault(obj, item, dflt):
    try:
        return obj[item]
    except (KeyError, IndexError):
        return dflt
    
def ircCardToBotCard(card):
    return card[0] + card[1].upper()

class DummyLogger:
    def info(self, *args):
        pass
