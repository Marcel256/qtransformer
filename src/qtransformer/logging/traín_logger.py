import sys
from abc import abstractmethod
import json

class Logger(object):

    @abstractmethod
    def log_metrics(self, metrics: dict):
        pass

    @abstractmethod
    def write_step(self):
        pass




class ConsoleLogger(Logger):

    def __init__(self):
        self.curr = dict()

    def log_metrics(self, metrics: dict):
        self.curr.update(metrics)

    def write_step(self):
        json.dump(self.curr, sys.stdout)
        self.curr = dict()
