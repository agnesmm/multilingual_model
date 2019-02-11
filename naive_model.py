import ipdb
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim

from data_loader import AmazonReviewLoader
from multilingual_embeddings import MultiLingualEmbeddings
from plot_scores import AccLossPlot
from trainer import Trainer


SEED = 0
torch.manual_seed(SEED)
random.seed(SEED)

class ConvNetwork(nn.Module):
    torch.set_default_tensor_type('torch.DoubleTensor')

    def __init__(self, nin, nout):
        super(ConvNetwork, self).__init__()

        self.conv1 = nn.Conv1d(nin, 1, kernel_size=2, stride=2, padding=0)
        self.max1 = nn.MaxPool1d(2)
        # self.conv2 = nn.Conv1d(10, 1, kernel_size=1, stride=2, padding=0)
        # self.max2 = nn.MaxPool1d(2)
        self.fc = nn.Linear(75, nout, bias=True)
        self.dropout = nn.Dropout(p=0.3)
        self.a = nn.Sigmoid()


    def forward(self, x):
        y = self.conv1(x)
        y = self.max1(y)
        # y = self.conv2(y)
        # y = self.max2(y)
        y=self.fc(y)
        y = self.dropout(y)
        y=self.a(y)
        y=y.view(y.shape[0],-1)
        return y

class RNN(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim=300,
                 hidden_dim=20, n_layers=2, bidirectional=True,
                 dropout=.2):
        super(RNN, self).__init__()
        # self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.a = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(1, 0, 2)
        #x = [sent len, batch size, emb dim]
        x = self.dropout(x)
        #output, hidden = self.rnn(x)
        output, (hidden, cell) = self.rnn(x)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        y = self.fc(hidden.squeeze(0))
        return y

class Model(Trainer):
    def __init__(self, source_languages, target_language,
                 n_epochs=100):

        self.nout = 1
        self.emb_shape = 300
        self.max_len = 200
        self.n_epochs = n_epochs
        self.source_languages = source_languages
        self.target_language = target_language

    def get_embedings(self, embeddings_dict):
        embeddings = embeddings_dict.embed
        embeddings = torch.Tensor(embeddings).type(torch.float)
        return nn.Embedding.from_pretrained(embeddings)

    def train_model(self, labeled_corpus, test, embeddings_dicts, model,
                    loss = nn.MSELoss(),
                    optimizer=optim.SGD, optim_param={},
                    model_name='CNN'):

        languages = self.source_languages + [self.target_language]
        iters = {l:iter(loader) for l, loader in labeled_corpus.items()}
        language_dict = {language:i for i, language in enumerate(labeled_corpus.keys())}
        embeddings = {l: self.get_embedings(embeddings_dicts[l]) for l in iters.keys()}

        train_plot = AccLossPlot(num=1, title='%s train'%model_name)
        test_plot = AccLossPlot(num=2, title='%s test'%model_name)

        self.model = model(self.max_len, self.nout)
        self.loss = loss
        self.opt = optimizer(self.model.parameters(), **optim_param)

        epoch = 0
        while epoch < self.n_epochs:
            total_loss_d = 0
            for _ in range(100):
                for language in self.source_languages:
                    iter_l = iters[language]
                    try:
                        x, y = next(iter_l)
                    except StopIteration:
                        iter_l = iter(labeled_corpus[language])
                        iters[language] = iter_l
                        x, y = next(iter_l)
                        epoch += 1
                    x = embeddings[language](x)
                    loss_train, acc_train = self.train(x.double(), y.double())
                    train_plot.update(loss_train, acc_train)

            for _, (x_t, y_t) in enumerate(test):
                with torch.no_grad():
                    x_t = embeddings[self.target_language](x_t)
                    loss_test, acc_test = self.validation(x_t.double(), y_t.double())

                test_plot.update(loss_test, acc_test)
                print('Test loss = %.3f. Test accuracy = %.3f'%(loss_test, acc_test))

        train_plot.update(loss_train, acc_train, save=True)
        test_plot.update(loss_test, acc_test, save=True)

if __name__ == '__main__':
    source_languages = ['fr', 'de'] #, 'de', 'jp'
    target_language = 'en'


    loader = AmazonReviewLoader(batch_size=32,
                                source_languages=source_languages,
                                target_language=target_language,
                                xml_limit_size=None)

    labeled_corpus, unlabeled_corpus, test, embeddings_dicts = loader.load_corpus()

    model = Model(n_epochs=10000,
                          source_languages=source_languages,
                          target_language=target_language)

    model.train_model(labeled_corpus, test, embeddings_dicts,
                RNN,
                optimizer=optim.Adam,
                optim_param={'lr' : 10e-3},
                model_name='test')
