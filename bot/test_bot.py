
from unittest import TestCase
from PokerBot import PokerBot


class TestBot(TestCase):


    def test1(self):
        bot = PokerBot(pokerSite="PS2")
        bot.start("savedScreenshots/test1.png")
        self.assertEqual(set(bot.getGameState()["myCards"][0]), {'TD','KS'})
        self.assertEqual(bot.getGameState()["currentCallValue"][0],300)

    def test2(self):
        bot = PokerBot(pokerSite="PS_old")
        bot.start("savedScreenshots/test2.png")
        self.assertEqual(set(bot.getGameState()["myCards"][0]), {'5S','QD'})
        
    def test3(self):
        bot = PokerBot(pokerSite="PS2")
        bot.start("savedScreenshots/test3.png")
        self.assertEqual(set(bot.getGameState()["tableCards"][0]), {'7H','5C','3C'})
        self.assertEqual(bot.getGameState()["currentCallValue"][0],0)

    def test4(self):
        bot = PokerBot(pokerSite="PS_old")
        bot.start("savedScreenshots/test4.png")
        self.assertEqual(set(bot.getGameState()["tableCards"][0]), {'9H','4H','QD','7S'})
        