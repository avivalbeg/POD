"""This script harvests useful info out of logged poker games 
and stores it in readable format."""
from mining.DataMiner import IrcDataMiner



if __name__ == '__main__':
    dc = IrcDataMiner()
    dc.mineHandData()
