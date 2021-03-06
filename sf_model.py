import ipdb
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.utils.data
import torch.nn as nn
import torch.optim as optim

from data_loader import AmazonReviewLoader
from multilingual_embeddings import MultiLingualEmbeddings
from plot_scores import AccLossPlot
from utils import timeit
from trainer import Trainer

SEED = 0
torch.manual_seed(SEED)
random.seed(SEED)
torch.set_default_tensor_type('torch.DoubleTensor')



class SharedFeatureExtractor(nn.Module):
    def __init__(self, emb_size, nout=128):
        super(SharedFeatureExtractor, self).__init__()
        self.hidden_size = 200
        self.fc1 = nn.Linear(emb_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, nout)

    def forward(self, x):
        y = F.relu(self.fc1(x))
        y = nn.Dropout(.5)(y)
        y = nn.BatchNorm1d(self.hidden_size)(y)
        y = F.relu(self.fc2(y))
        y = nn.Dropout(.5)(y)
        y = nn.BatchNorm1d(self.hidden_size)(y)
        return y


class LanguageDiscriminator(nn.Module):
    def __init__(self, seq_len, nout, dropout=.5):
        super(LanguageDiscriminator, self).__init__()
        nhidden=300
        self.net = nn.Sequential(nn.Linear(seq_len*128, nhidden),
                                 nn.ReLU(),
                                 nn.BatchNorm1d(nhidden),
                                 nn.Dropout(dropout),

                                 nn.Linear(nhidden, nhidden),
                                 nn.ReLU(),
                                 nn.BatchNorm1d(nhidden),
                                 nn.Dropout(dropout),

                                 nn.Linear(nhidden, nout),
                                 nn.Softmax())

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.net(x)


class Classifier(nn.Module):
    def __init__(self, seq_len, sf_size, nout=2):
        super(Classifier, self).__init__()
        hidden_size=64
        n_layers=2
        bidirectional=False
        dropout=.5
        # self.rnn = nn.RNN(embedding_dim, hidden_size)
        self.rnn = nn.LSTM(sf_size, hidden_size,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)
        self.bn = nn.BatchNorm1d(hidden_size*2)
        self.fc = nn.Linear(hidden_size*2, nout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x : batch_size x seq_len x sf_size
        x = x.transpose(0,1)
        # x : x seq_len x batch_size  x sf_size
        x = self.dropout(x)
        #output, hidden = self.rnn(x)
        output, (hidden, cell) = self.rnn(x)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        hidden = self.bn(hidden)
        y = F.softmax(self.fc(hidden.squeeze(0)))
        return y



class Model(Trainer):
    def __init__(self, source_languages, target_language, d_iter=1, n_epochs=1000):
        self.nout = 2
        self.emb_size = 300
        self.sf_size = 128
        self.seq_len = 200 #todo
        self.n_languages = len(source_languages) + 1
        self.d_iter = d_iter
        self.source_languages = source_languages
        self.target_language = target_language
        self.n_epochs = n_epochs



    def get_embedings(self, embeddings_dict):
        embeddings = embeddings_dict.embed
        embeddings = torch.Tensor(embeddings).type(torch.float)
        return nn.Embedding.from_pretrained(embeddings)

    def train(self, labeled_corpus, unlabeled_corpus, test, embeddings_dicts):


        emb_size = self.emb_size
        sf_size = self.sf_size
        seq_len = self.seq_len
        nout = self.nout
        languages = self.source_languages + [self.target_language]
        lr = 0.001
        weight_decay = 10e-8

        train_plot = AccLossPlot(num=3, title='shared features train')
        test_plot = AccLossPlot(num=4, title='shared features test')
        n_languages = len(list(labeled_corpus.keys()))


        FS = SharedFeatureExtractor(emb_size, sf_size)


        D = LanguageDiscriminator(seq_len, n_languages).double()
        D_loss = nn.NLLLoss()
        D_opt = optim.Adam(D.parameters(), lr=lr)

        C = Classifier(seq_len=seq_len, sf_size=sf_size, nout=nout)
        C_loss = nn.NLLLoss()

        c_optim = optim.Adam(list(C.parameters()) + list(FS.parameters()),
                             lr=lr, weight_decay=weight_decay)
        def accuracy(ypred, ytrue):
            return float(ypred.eq(ytrue).sum())/float(len(ytrue))

        language_dict = {language:i for i, language in enumerate(labeled_corpus.keys())}
        test_iter = iter(test)
        labeled_iters = {l:iter(loader) for l, loader in labeled_corpus.items()}
        unlabeled_iters = {l:iter(loader) for l, loader in unlabeled_corpus.items()}
        embeddings = {l: self.get_embedings(embeddings_dicts[l]) for l in languages}

        for i in range(self.n_epochs):

            FS.train()
            C.train()
            D.train()
            ###### train D ######
            d_accuracy = []
            for _ in range(self.d_iter):
                total_loss_d = 0
                D.zero_grad()
                for language in languages:
                    language_index = language_dict[language]
                    iter_l = unlabeled_iters[language]

                    # endless iter
                    try:
                        x, y = next(iter_l)
                    except StopIteration:
                        iter_l = iter(unlabeled_corpus[language])
                        unlabeled_iters[language] = iter_l
                        x, y = next(iter_l)

                    x = embeddings[language](x).double() # batch_size x seq_len x emb_size
                    shared_features = FS(x) # batch_size x seq_len x sf_size
                    true_l = torch.Tensor([language_index]).expand(len(x), 1).long()
                    predicted_l = D(shared_features)
                    d_loss = D_loss(predicted_l, true_l.view(-1))
                    d_accuracy.append(accuracy(predicted_l.max(1)[1], true_l.view(-1)))
                    total_loss_d += d_loss
                total_loss_d.backward()
                D_opt.step()
            print('D accuracy', np.mean(d_accuracy))

            ###### train FS and C ######
            total_loss = 0
            FS.zero_grad()
            C.zero_grad()
            for language in self.source_languages:
                language_index = language_dict[language]
                iter_l = labeled_iters[language]

                try:
                    x, y = next(iter_l)
                except StopIteration:
                    iter_l = iter(labeled_corpus[language])
                    labeled_iters[language] = iter_l
                    x, y = next(iter_l)

                x = embeddings[language](x).double() # batch_size x seq_len x emb_size
                shared_features = FS(x) # batch_size x seq_len x sf_size
                ypred = C(shared_features) # batch_size x nout
                c_loss = C_loss(ypred, y.view(-1))
                c_accuracy = accuracy(ypred.max(1)[1], y.view(-1))
                print('%s Train loss = %.3f. Train accuracy = %.3f'%(language, float(c_loss), c_accuracy))
                train_plot.update(float(c_loss), float(c_accuracy))
                total_loss += c_loss


            for language in languages:
                language_index = language_dict[language]
                iter_l = unlabeled_iters[language]

                try:
                    x, y = next(iter_l)
                except StopIteration:
                    iter_l = iter(unlabeled_corpus[language])
                    unlabeled_iters[language] = iter_l
                    x, y = next(iter_l)

                x = embeddings[language](x).double()
                shared_features = FS(x)
                predicted_l = D(shared_features)
                true_l = torch.Tensor([language_index]).expand(len(x), 1).long()
                d_loss = -D_loss(predicted_l, true_l.view(-1))
                total_loss += 0.002*d_loss

            total_loss.backward()
            c_optim.step()


            FS.eval()
            C.eval()
            D.eval()
            try:
                x_t, y_t = next(test_iter)
            except StopIteration:
                test_iter = iter(test)
                x_t, y_t = next(test_iter)
            with torch.no_grad():
                x_t = embeddings[self.target_language](x_t).double()
                pred_t = C(FS(x_t))
                loss_test = float(C_loss(pred_t, y_t.view(-1)))
                acc_test =  accuracy(pred_t.max(1)[1], y_t.view(-1))
                test_plot.update(loss_test, acc_test)
                print('Test loss = %.3f. Test accuracy = %.3f'%(loss_test, acc_test))



        train_plot.update(float(c_loss), float(c_accuracy), save=True)
        test_plot.update(float(loss_test), float(acc_test), save=True)


# todo mentor toutes les loss
#


if __name__ == '__main__':

    source_languages = ['fr', 'en', 'jp']
    target_language = 'de'

    loader = AmazonReviewLoader(batch_size=16,
                                source_languages=source_languages,
                                target_language=target_language,
                                xml_limit_size=None)

    labeled_corpus, unlabeled_corpus, test, embeddings_dicts = loader.load_corpus()

    model = Model(source_languages=source_languages,
                  target_language=target_language,
                  n_epochs=10000)


    model.train(labeled_corpus, unlabeled_corpus, test, embeddings_dicts)
