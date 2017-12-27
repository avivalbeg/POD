"""
This module contains various methods for doing statistical computations. 
Unless noted otherwise, the names of the method correspond to the statistical functions with the same names.

I excessively convert numbers into floats in order to prevent any chance of 
accidentally dividing by an integer, which could create a difficult bug to detect.
"""


from cmath import sqrt
import random
from _collections import defaultdict
from random import choice
from pprint import pprint
from scipy.stats.mstats_basic import Ttest_1sampResult
from scipy.stats.stats import chisquare

__all__ = ["mean", "distance_from_mean", "squared_distance_from_mean",
           "ss", "ms", "var", "sd", "z_scores", "r", "sum_mul", "mm", "sum_sq",
           "ClassicalLinearModel"]

ROUND_N = 1000  # This determines the decimal precision


def div_or_0(x,y):
    return x/y

def floats(X):
    """
    Convert a vector to floats.
    """
    return [float(x) for x in X]

def roundN(x):
    return round(float(x), ROUND_N)

def sum(X):
    s = 0
    for x in X: s += x
    return float(s)

def mean(X):
    return roundN(float(sum(X)) / float(len(X)))

def distance_from_mean(X):
    m = mean(X)
    return [roundN(x - m) for x in X]

def squared_distance_from_mean(X):
    return [roundN(x ** 2) for x in distance_from_mean(X)]

def ss(X):
    return (sum(squared_distance_from_mean(X)))

def meanStandardError(X):
    m=mean(X)
    return mean([(m-x)**2 for x in X])/sd(X)

def ms(X): 
    """
    Mean of squares.
    """
    return div_or_0(sum([x ** 2 for x in X]),
                    len(X))


def var(X):
    return roundN(ss(X) / float(len(X)))

def sd(X):
    return roundN(sqrt(var(X)).real)

def z_scores(X):
    """
    Returns z-scores for all values in X.
    """
    this_sd = sd(X)
    return [roundN(div_or_0(val , this_sd)) for val in distance_from_mean(X)]

def r(X, Y):
    if len(X) != len(Y): raise IOError
    N = float(len(X))
    zs1 = z_scores(X)
    zs2 = z_scores(Y)
    return round(sum(zs1[i] * zs2[i] for i in range(int(N))) / N, 2)


def sum_muls(X, Y):
    """
    Sum of multiples of X and Y: \Sigma X_i*Y_i.
    """
    if len(X) != len(Y): raise IOError
    N = float(len(X))
    return float(sum([X[i] * Y[i] for i in range(len(X))]))

def mm(X, Y):
    """
    Mean of multiples: (\Sigma X_i*Y_i)/N.
    """
    if len(X) != len(Y): raise IOError
    N = float(len(X))
    return div_or_0(sum_muls(X, Y), N)

def sum_sq(X):
    """
    Sum of squares. Notice that this is not the sum of squared distance from the mean.
    """
    return float(sum([x ** 2 for x in X]))


# Models


class StatisticalModel:
    pass

class RegressionModel(StatisticalModel):
    pass

class LinearModel(RegressionModel):
    def __init__(self, X, Y):
        if len(X) != len(Y): raise IOError
        
        ssq = sum_sq(X)
        sX = sum(X)
        sY = sum(Y)
        sMul = sum_muls(X, Y)
        sXsq = sX ** 2
        
        self.N = len(X)
        self.X = X
        self.Y = Y
        self.slope = div_or_0(self.N * sMul - sX * sY, self.N * ssq - sXsq)
        self.intercept = div_or_0((sY * ssq - sX * sMul), self.N * ssq - sXsq)
        self.model = lambda x: self.intercept + self.slope * x
        
        self.model_variance = sqrt(div_or_0(sum([(Y[i] - self.model(X[i])) ** 2 for i in range(self.N)]),
                             self.N - 2)).real
        
        self.intercept_se = self.model_variance * sqrt(div_or_0(1, self.N) + div_or_0(mean(X) ** 2, ss(X))).real    
        self.slope_se = div_or_0(self.model_variance,
                    sqrt(ss(X)).real)
        self.t_value = abs(div_or_0(self.slope, self.slope_se))
        self.r_sq = r(X, Y) ** 2
    
        self.margin_of_error_slope = lambda critical_value: critical_value * self.slope_se
        self.margin_of_error_intercept = lambda critical_value: critical_value * self.slope_se
    


T=10
J=11
Q=12
K=13
A=14

ppDist ={'2C': 347, '2D': 322, '2H': 343, '2S': 322, '3C': 323, '3D': 310, '3H': 315, '3S': 325, '4C': 348, '4D': 353, '4H': 357, '4S': 367, '5C': 368, '5D': 377, '5H': 413, '5S': 392, '6C': 417, '6D': 386, '6H': 405, '6S': 401, '7C': 407, '7D': 451, '7H': 475, '7S': 453, '8C': 413, '8D': 433, '8H': 446, '8S': 431, '9C': 453, '9D': 466, '9H': 492, '9S': 475, 'AC': 635, 'AD': 630, 'AH': 641, 'AS': 655, 'JC': 531, 'JD': 572, 'JH': 564, 'JS': 489, 'KC': 607, 'KD': 616, 'KH': 591, 'KS': 605, 'QC': 525, 'QD': 573, 'QH': 535, 'QS': 524, 'TC': 483, 'TD': 482, 'TH': 516, 'TS': 526}
psDist ={'2C': 95, '2D': 124, '2H': 121, '2S': 109, '3C': 131, '3D': 127, '3H': 106, '3S': 133, '4C': 118, '4D': 110, '4H': 145, '4S': 114, '5C': 120, '5D': 116, '5H': 136, '5S': 126, '6C': 143, '6D': 134, '6H': 125, '6S': 131, '7C': 151, '7D': 138, '7H': 124, '7S': 128, '8C': 140, '8D': 138, '8H': 134, '8S': 142, '9C': 106, '9D': 138, '9H': 126, '9S': 150, 'AC': 172, 'AD': 190, 'AH': 169, 'AS': 186, 'JC': 146, 'JD': 144, 'JH': 133, 'JS': 129, 'KC': 168, 'KD': 158, 'KH': 166, 'KS': 179, 'QC': 151, 'QD': 146, 'QH': 159, 'QS': 146, 'TC': 137, 'TD': 149, 'TH': 162, 'TS': 127}
ppDistTable = {'2C': 261, '2D': 212, '2H': 255, '2S': 256, '3C': 249, '3D': 243, '3H': 262, '3S': 250, '4C': 267, '4D': 247, '4H': 228, '4S': 271, '5C': 280, '5D': 277, '5H': 261, '5S': 266, '6C': 256, '6D': 242, '6H': 231, '6S': 264, '7C': 256, '7D': 252, '7H': 244, '7S': 246, '8C': 268, '8D': 269, '8H': 230, '8S': 257, '9C': 234, '9D': 255, '9H': 244, '9S': 239, 'AC': 235, 'AD': 224, 'AH': 250, 'AS': 243, 'JC': 256, 'JD': 239, 'JH': 219, 'JS': 262, 'KC': 248, 'KD': 270, 'KH': 240, 'KS': 241, 'QC': 236, 'QD': 239, 'QH': 242, 'QS': 255, 'TC': 232, 'TD': 264, 'TH': 279, 'TS': 243}

scrapedDist = defaultdict(lambda:0)
with open("../bot/cardsLog.txt") as f:
    for line in f:
        if line:
            card = line.strip().split(" ")[0]
            denom = card[0]
            scrapedDist[denom]+=1

def makesimDist(n, keys):
    dist=defaultdict(lambda:0)
    for _ in range(int(n)):
        dist[choice(keys)]+=1
    return dict(dist)

def evalFairness(cardDist):
    data1 = [(eval(card[0]),count) for card,count in cardDist.items()]
    total = sum([count for denom, count in data1])
    
    simDist = makesimDist(total, list(cardDist.keys()))
    
    data2 = [(eval(card[0]),count) for card,count in simDist.items()]

    reg1 = LinearModel([denom for denom,count in data1],[count for denom,count in data1])
    reg2 = LinearModel([denom for denom,count in data2],[count for denom,count in data2])
    print("Fairness evalutaion for card denomination distribution with",int(sum(list(scrapedDist.values()))), "observations")
    print("------------------------------")
    print("Simulated stats:")
    print("------------------------------")
    print("Simulated mean: ",int(mean(list(simDist.values()))))
    print("Simulated sd:",int(sd(list(simDist.values()))))
    print("Simulated var:",int((var(list(simDist.values())))))
    print()
    print("Linear regression: ")
    print("Simulated slope:",int(reg2.slope))
    print("Simulated intercept:",int(reg2.intercept))
    print("Simulated rsq:",(reg2.r_sq))
    print("Simulated t value:",(reg2  .t_value))
    print()

    print("Analytic stats (for multinomial distribution with P(Xi)=1/13):")
    print("------------------------------")
    n = sum(list(scrapedDist.values()))
    p = 1./len(scrapedDist)
    E = n*p
    theVar = E*(1-p)
    theSd = roundN(sqrt(theVar).real)
    
    print("Multinomial mean:",  int(E))
    print("Multinomial sd:", int(theSd))
    print("Multinomial var:",int(theVar))
    print()
    print("Observed stats:")
    print("------------------------------")
    print("Observed mean:",int(mean(list(cardDist.values()))))
    print("Observed sd:",int(sd(list(cardDist.values()))))
    print("Observed var:", int((var(list(cardDist.values())))))
    print()
    print("Linear regression:")
    print("Observed slope:" ,reg1.slope)
    print("Observed intercept:",int(reg1.intercept))
    print("Observed rsq:",reg1.r_sq)
    print("Obseved t value:" , reg1.t_value)
    print()
    print("Chi-sq test:")
    div,p_value = chisquare(list(cardDist.values()))
    print("chi-sq:", div)
    print("p=",p,sep="")
    print()
    print("Chi-sq test for simulated distribution:")
    div,p_value = chisquare(list(simDist.values()))
    print("chi-sq:", div)
    print("p=",p,sep="")
    print()
    print("Critical value for chi-sq test with",len(cardDist)-1,"degrees of freedom:")
    print(5.22603)
    

    
    
evalFairness(scrapedDist)


quit()
n = sum(list(ppDistTable.values()))
keys = list(ppDistTable.keys())
fairs = [var(list(makeSimDist(n, keys).values())) for _ in range(1000)]
evalFairness(ppDistTable)
evalFairness(ppDist)
# evalFairness(psDist)
    
    
    
    
                
