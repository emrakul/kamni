from argparse import ArgumentError
from navec import Navec
from scipy.spatial.distance import cosine
import tqdm
import numpy as np

class Bot():
    def __init__(self, secret=None):
        path='hudlit_12B_500K_300d_100q.tar'
        self.en_alphabet = [chr(ord('a')+i) for i in range(0,26)] 
        self.ru_alphabet = [chr(ord('а')+i) for i in range(0,32)]  + ['ё']
        self.model = Navec.load(path)
        self.start(secret)

    def detect_language(self, word) -> str:
        if all([c in self.en_alphabet for c in word]):
            return 'en'
        else:
            return 'ru'


    def start(self, secret: str):

        self.secret = secret
        self.lang = self.detect_language(secret)
        self.emb = self.model[self.secret]
        self.vocab = list(filter(lambda x: 
            self.detect_language(x) == self.lang, self.model.vocab.words))
        self.secret = secret
        self.emb = self.model[self.secret]
        # generation
        self.top_words = sorted(self.vocab, 
            key=lambda x: 
            cosine(self.model[x], self.emb))[:1000]
        
    def printout(self, distance, top=None) -> str:
        rounded = np.round(distance, 3) 
        if top is not None:
            progress = tqdm.tqdm.format_meter(top, 1000, 500)
            return str(rounded) + " "  +progress[progress.index('%')+1:progress.index('[')]
        else:
            return str(rounded)

    def guess(self, guess: str) -> str:
        if guess not in self.vocab:
            return "такого слова НЕТ"
        if guess == self.secret:
            return "УГАДАНО"
        vector = self.model[guess]
        if guess in self.top_words:
            distance = cosine(vector, self.emb) 
            rank = self.top_words.index(guess)
            return self.printout(distance, rank)
        else:
            distance = cosine(vector, self.emb)
            return self.printout(distance)
