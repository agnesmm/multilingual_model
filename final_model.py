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



class Classifier(nn.Module):
    def __init__(self, seq_len, nout, sf_size, pf_size, dropout=.5):
        super(Classifier, self).__init__()
        nhidden=300
        self.net = nn.Sequential(nn.Linear(seq_len*(sf_size+pf_size), nhidden),
                                 nn.ReLU(),
                                 nn.BatchNorm1d(nhidden),
                                 nn.Dropout(dropout),

                                 # nn.Linear(nhidden, nhidden),
                                 # nn.ReLU(),
                                 # nn.BatchNorm1d(nhidden),
                                 # nn.Dropout(dropout),

                                 nn.Linear(nhidden, nout),
                                 nn.Softmax())

    def forward(self, sf, pf):
        x = torch.cat((sf, pf), dim=2)
        x = x.view(x.shape[0], -1)
        return self.net(x)

class LanguageDiscriminator(nn.Module):
    def __init__(self, seq_len, nout, dropout=.5):
        super(LanguageDiscriminator, self).__init__()
        nhidden=300
        self.net = nn.Sequential(nn.Linear(seq_len*128, nhidden),
                                 nn.ReLU(),
                                 nn.BatchNorm1d(nhidden),
                                 nn.Dropout(dropout),

                                 # nn.Linear(nhidden, nhidden),
                                 # nn.ReLU(),
                                 # nn.BatchNorm1d(nhidden),
                                 # nn.Dropout(dropout),

                                 nn.Linear(nhidden, nout),
                                 nn.Softmax())

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.net(x)


class SharedFeatureExtractor(nn.Module):
    def __init__(self, emb_size=300, sf_size=128, hidden_dim=64, dropout=.5):
        super(SharedFeatureExtractor, self).__init__()
        self.lstm = nn.LSTM(emb_size, hidden_dim, num_layers=1,
                            bidirectional=True, dropout=dropout)

    def forward(self, x):
        x = x.double()
        output, (hidden, cell) = self.lstm(x)
        return output

class PrivateFeaturesExtractor(nn.Module):
    def __init__(self, seq_len, emb_size, n_sources, pf_size, hidden_dim=64, dropout=.5):
        super(PrivateFeaturesExtractor, self).__init__()
        hidden_size = 200
        self.pf_size = pf_size
        self.n_sources=n_sources
        self.pf_extractors = nn.ModuleList([nn.Sequential(
                                nn.LSTM(emb_size, hidden_dim,
                                        num_layers=1,
                                        bidirectional=True,
                                        dropout=dropout)) for _ in range(n_sources)])

        self.g = nn.Sequential( nn.Linear(emb_size, hidden_size),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        nn.BatchNorm1d(hidden_size),
                                        nn.Linear(hidden_size, n_sources),
                                        nn.Dropout(dropout),
                                        nn.BatchNorm1d(hidden_size),
                                        nn.Softmax(dim=-1))



    def forward(self, x):
        privates_features = [net(x)[0] for net in self.pf_extractors]
        x0 = x.shape[0]
        x1 = x.shape[1]
        gates = self.g(x)

        gates_coeff = gates.unsqueeze(-2).expand(x0, x1, self.pf_size, self.n_sources)
        output = torch.stack(privates_features, dim=-1)

        output = gates_coeff*output
        output = output.sum(dim=-1)
        return gates, output




class Model(Trainer):
    def __init__(self, source_languages, target_language, d_iter=1):
        self.nout = 2
        self.emb_size = 300
        self.sf_size = 128
        self.pf_size = 128
        self.seq_len = 200 #todo
        self.n_languages = len(source_languages) + 1
        self.d_iter = d_iter
        self.source_languages = source_languages
        self.target_language = target_language


    def get_embedings(self, embeddings_dict):
        embeddings = embeddings_dict.embed
        embeddings = torch.Tensor(embeddings).type(torch.float)
        return nn.Embedding.from_pretrained(embeddings)

    def train(self, labeled_corpus, unlabeled_corpus, test, embeddings_dicts):

        lambda_1 = 0.002
        lambda_2 = 0.1
        emb_size = self.emb_size
        sf_size = self.sf_size
        pf_size = self.pf_size
        seq_len = self.seq_len
        nout = self.nout
        lr = 0.001
        weight_decay = 10e-8
        languages = self.source_languages + [self.target_language]

        train_plot = AccLossPlot(num=1, title='Train')
        test_plot = AccLossPlot(num=2, title='Test')
        n_languages = len(list(labeled_corpus.keys()))


        FS = SharedFeatureExtractor(emb_size=emb_size, sf_size=sf_size)

        D = LanguageDiscriminator(seq_len, n_languages).double()
        D_loss = nn.NLLLoss()
        D_opt = optim.Adam(D.parameters(), lr=lr, weight_decay=weight_decay)

        FP = PrivateFeaturesExtractor(seq_len = seq_len,
                                     emb_size = emb_size,
                                     n_sources = len(self.source_languages),
                                     pf_size = self.pf_size)


        C = Classifier(seq_len=seq_len, nout=nout, sf_size=sf_size, pf_size=pf_size)
        C_loss = nn.NLLLoss()
        c_optim = optim.Adam(list(C.parameters()) + list(FS.parameters()) + list(FP.parameters()),
                           lr=lr, weight_decay=weight_decay)

        G_loss = nn.NLLLoss()

        def accuracy(ypred, ytrue):
            return float(ypred.eq(ytrue).sum())/float(len(ytrue))

        language_dict = {language:i for i, language in enumerate(labeled_corpus.keys())}
        test_iter = iter(test)
        labeled_iters = {l:iter(loader) for l, loader in labeled_corpus.items()}
        unlabeled_iters = {l:iter(loader) for l, loader in unlabeled_corpus.items()}
        embeddings = {l: self.get_embedings(embeddings_dicts[l]) for l in languages}

        for i in range(100):
            FP.train()
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
            FP.zero_grad()
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
                gates, privates_features = FP(x)
                print('TRAIN', language, gates.mean(dim=0).mean(dim=0))
                true_l = torch.Tensor([language_index]).expand(len(x)*self.seq_len, 1).long()
                print('TRAIN truth', language, true_l.view(-1).double().mean(dim=0)
                    )

                gates = gates.view(gates.shape[0]*gates.shape[1], gates.shape[2])
                ypred = C(shared_features, privates_features) # batch_size x nout

                c_loss = C_loss(ypred, y.view(-1))
                c_accuracy = accuracy(ypred.max(1)[1], y.view(-1))
                print('%s Train loss = %.3f. Train accuracy = %.3f'%(language, float(c_loss), c_accuracy))
                train_plot.update(float(c_loss), float(c_accuracy))
                g_loss = G_loss(gates, true_l.view(-1))
                total_loss += c_loss + lambda_2*g_loss



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
                total_loss += -lambda_1*D_loss(predicted_l, true_l.view(-1))

            total_loss.backward()
            c_optim.step()


            FP.eval()
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
                gates, fp_t = FP(x_t)
                print('TEST', language, gates.mean(dim=0).mean(dim=0))
                pred_t = C(FS(x_t), fp_t)
                loss_test = float(C_loss(pred_t, y_t.view(-1)))
                acc_test =  accuracy(pred_t.max(1)[1], y_t.view(-1))
                test_plot.update(loss_test, acc_test)
                print('%d Test loss = %.3f. Test accuracy = %.3f'%(i, loss_test, acc_test))



        train_plot.update(float(c_loss), float(c_accuracy), save=True)
        test_plot.update(float(loss_test), float(acc_test), save=True)




if __name__ == '__main__':

    source_languages = ['fr', 'en', 'jp']
    target_language = 'de'

    loader = AmazonReviewLoader(batch_size=16,
                                source_languages=source_languages,
                                target_language=target_language,
                                xml_limit_size=500)



    labeled_corpus, unlabeled_corpus, test, embeddings_dicts = loader.load_corpus()

    model = Model(source_languages=source_languages,
                  target_language=target_language)


    model.train(labeled_corpus, unlabeled_corpus, test, embeddings_dicts)
