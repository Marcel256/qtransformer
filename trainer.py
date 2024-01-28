from abc import abstractmethod

from qtransformer import QTransformer
from sequence_dataset import SequenceDataset

class Trainer(object):

    @abstractmethod
    def train(self, model: QTransformer, dataset: SequenceDataset):
        pass
