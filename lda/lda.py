import numpy as np
import scipy
from tqdm import tqdm
from scipy.special import digamma

class LDA:

    def __init__(self, K, documents):
        self.num_topics = K
        self.documents = documents
        self._init()


    def _init(self):
        corpus = set()
        num_words = 0
        for doc in self.documents:
            corpus |= doc
            num_words += len(doc)

        self.corpus = corpus
        self.word_to_idx = {word: idx for (idx, word) in enumerate(self.corpus)}
        self.idx_to_word = {idx: word for (word, idx) in self.word_to_idx.items()}
        self.i_documents = [[self.word_to_idx[word] for word in doc] for doc in self.documents]

        self.num_words = num_words
        self.num_docs = len(self.documents)
        self.num_corpus = len(self.corpus)
        self.num_documents = len(self.documents)

        self.alpha = 50 / self.num_topics
        self.phi = [
            np.ones(shape=(len(doc), self.num_topics)) * (1 / self.num_topics) for doc in self.i_documents
        ]

        self.gamma = [
            np.ones(shape=(self.num_topics,)) * (self.alpha + (len(doc) / self.num_topics)) for doc in self.i_documents
        ]
        self.beta = np.random.random(size=(self.num_topics, self.num_words))
        for i in range(self.beta.shape[0]):
            self.beta[i] /= self.beta[i].sum()



    def train(self, num_step):
        for i in tqdm(range(num_step)):
            self._E_step()
            print(self.phi[0])
            self._M_step()


    def _E_step(self):
        for i in range(5):
            for d in range(self.num_docs):
                words = self.i_documents[d]
                dv = digamma(np.sum(self.gamma[d]))
                for n in range(len(words)):
                    self.phi[d][n] = self.beta[:, words[n]] * np.exp(digamma(self.gamma[d]) - dv)
                    self.phi[d][n] /= (1e-6 + np.sum(self.phi[d][n]))
                self.gamma[d] = self.alpha + np.sum(self.phi[d], axis=0)

            # TODO: check convergence, not iterate

    def _M_step(self):
        for _ in range(3):
            for d in range(self.num_docs):
                phi = self.phi[d]
                words = self.i_documents[d]
                for i in range(self.num_topics):
                    self.beta[i][words] += phi[:, i]
            for i in range(self.num_topics):
                self.beta[i] /= (1e-6 + np.sum(self.beta[i]))
        # Omit updating alpha. its very tough...

if __name__ == "__main__":
    documents = []
    with open("../data/ml-1m/stream", 'r') as f:
        i = 0
        while True:
            l = f.readline()
            if l == "":
                break
            documents.append(set(l.strip().split(' ')))
            i += 1
            if i == 2000:
                break

    lda = LDA(10, documents)
    lda.train(3)
