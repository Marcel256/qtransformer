from abc import abstractmethod


class Logger(object):

    @abstractmethod
    def log(self, metrics: dict):
        pass



class NoLogger(Logger):
    def log(self, metrics: dict):
        # Do nothing
        pass


class CsvLogger(Logger):

    def __init__(self, file):
        self.file = file




class ConsoleLogger(Logger):

    def __init__(self):
        pass

    def log(self, metrics: dict):
        for key in metrics.keys():
            print(key, ": ", metrics[key])