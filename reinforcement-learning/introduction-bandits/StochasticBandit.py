import numpy as np
import Arms as arms

class MAB:
    def __init__(self,arms):
        """given a list of arms, create the MAB environnement"""
        self.arms = arms
        self.nbArms = len(arms)
        self.means = [arm.mean for arm in arms]
        self.bestarm = np.argmax(self.means)
    
    def generateReward(self,arm):
        return self.arms[arm].sample()

## some functions that create specific MABs

def BernoulliBandit(means):
    """define a Bernoulli MAB from a vector of means"""
    return MAB([arms.Bernoulli(p) for p in means])


