from abc import ABC, abstractmethod

class BasicFitness(ABC): 
    @abstractmethod
    def __call__(self, population, config):
        pass