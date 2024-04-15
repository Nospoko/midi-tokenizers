from midi_tokenizers.midi_tokenizer import MidiTokenizer
from abc import abstractmethod
from datasets import Dataset
import tempfile
from tokenizers import trainers, models, Tokenizer


class MidiTrainableTokenizer(MidiTokenizer):
    def __init__(self):
        self.base_tokenizer: MidiTokenizer = None
        self.tokenizer: Tokenizer = None
        self.trainer: trainers.Trainer = None
    
    @abstractmethod
    def prepare_data_for_training(file_name: str, train_dataset: Dataset):
        pass
    
    def train(self, train_dataset: Dataset):
        file = tempfile.NamedTemporaryFile()
        self.prepare_data_for_training(file_name=file.name, train_dataset=train_dataset)

        trainer = trainers.BpeTrainer(max_token_length=512, special_tokens=["<CLS>"])
        self.tokenizer.model = models.BPE()
        self.tokenizer.train([file.name], trainer=trainer)