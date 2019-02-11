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


'''
  * sf : lstm
  * pf : lstm
  * d : 2 layers MLP
  * c : cnn
'''

# checker ce que ressort des gates
# normalise pour = 1
#

class SharedFeatureExtractor(nn.Module):
    def __init__(self, emb_size, nout=128):
        super(SharedFeatureExtractor, self).__init__()
        self.hidden_size = 200
        self.fc1 = nn.Linear(emb_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, nout)

    def forward(self, x):
        y = F.relu(self.fc1(x))
        y = nn.Dropout(.2)(y)
        y = nn.BatchNorm1d(self.hidden_size)(y)
        y = F.relu(self.fc2(y))
        y = nn.Dropout(.2)(y)
        y = nn.BatchNorm1d(self.hidden_size)(y)
        return y


class LanguageDiscriminator(nn.Module):
    def __init__(self, seq_len, nout):
        super(LanguageDiscriminator, self).__init__()
        self.conv = nn.Conv1d(seq_len, 5, kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(5, nout)

    def forward(self, x):
        x = x.double() # batch_size x seq_len x sf_size
        y = self.conv(x) #batchsize x seq_len x ?
        y = F.max_pool1d(y, y.shape[2]) #batchsize x seq_len
        y = y.squeeze(2)
        y = self.dropout(y)
        y = self.fc(y) #batchsize x nout
        y = y.view(y.shape[0],-1)
        return F.softmax(y) #batch_size x nout

class Classifier2(nn.Module):
    def __init__(self, seq_len, sf_size, nout=2, nhidden=64):
        super(Classifier, self).__init__()

        D = sf_size
        C = 2
        Ci = 1
        Co = 100
        Ks = (3,4,5)

        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(.2)
        self.fc1 = nn.Linear(len(Ks)*Co, C)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        y = self.fc1(x)
        return y




# class Classifier(nn.Module):
#     def __init__(self, seq_len, sf_size, nout=2):
#         super(Classifier, self).__init__()
#         #self.conv = nn.Conv1d(seq_len, 5, kernel_size=2, stride=2, padding=0)
#         self.dropout = nn.Dropout(p=0.3)
#         self.fc = nn.Linear(5, nout)

#     def forward(self, sf, pf):
#         x = torch.cat((sf, pf), dim=2)
#         x = x.double() # batch_size x seq_len x sf_size
#         y = self.conv(x) #batchsize x seq_len x ?
#         y = F.max_pool1d(y, y.shape[2]) #batchsize x seq_len
#         y = y.squeeze(2)
#         y = self.dropout(y)
#         y = self.fc(y) #batchsize x nout
#         y = F.softmax(y.view(y.shape[0],-1))
#         return y #batch_size x nout


class PrivateFeaturesExtractor(nn.Module):
    def __init__(self, seq_len, emb_size, n_sources, pf_size):
        super(PrivateFeaturesExtractor, self).__init__()
        hidden_size = 200
        self.pf_size = pf_size
        self.pf_extractors = nn.ModuleList([nn.Sequential(nn.Linear(emb_size, hidden_size),
                                            nn.ReLU(),
                                            nn.Dropout(0.5),
                                            nn.BatchNorm1d(hidden_size),
                                            nn.Linear(hidden_size, pf_size),
                                            nn.ReLU(),
                                            nn.Dropout(0.5),
                                            nn.BatchNorm1d(hidden_size)) for _ in range(n_sources)])

        self.pf_gates = nn.ModuleList([nn.Sequential( nn.Linear(emb_size, hidden_size),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.BatchNorm1d(hidden_size),
                                        nn.Linear(hidden_size, 1),
                                        nn.Tanh(),
                                        nn.Dropout(0.5),
                                        nn.BatchNorm1d(hidden_size)) for _ in range(n_sources)])



    def forward(self, x):
        privates_features = [net(x) for net in self.pf_extractors]
        x0 = x.shape[0]
        x1 = x.shape[1]
        gates_forward = [net(x) for net in self.pf_gates]
        gates = torch.stack([g.squeeze(-1) for g in gates_forward])
        gates = gates.permute(1,2,0)
        gates_coeff = [g.expand(x0, x1, self.pf_size) for g in gates_forward]
        output = torch.stack([pf*gate for pf, gate in zip(privates_features, gates_coeff)])
        output = output.sum(dim=0)
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

        train_plot = AccLossPlot(num=1, title='private features train')
        test_plot = AccLossPlot(num=2, title='private features test')
        n_languages = len(list(labeled_corpus.keys()))


        FS = SharedFeatureExtractor(emb_size, sf_size)
        # FS_optim = optim.Adam(FS.parameters(), lr=lr)

        D = LanguageDiscriminator(seq_len, n_languages).double()
        D_loss = nn.NLLLoss()
        D_opt = optim.Adam(D.parameters(), lr=lr, weight_decay=weight_decay)

        FP = PrivateFeaturesExtractor(seq_len = seq_len,
                                     emb_size = emb_size,
                                     n_sources = len(self.source_languages),
                                     pf_size = self.pf_size)

        # FP_optim = optim.Adam(FP.parameters(), lr=lr)

        C = Classifier(seq_len=seq_len, sf_size=sf_size, nout=nout)
        C_loss = nn.NLLLoss()
        # optim.Adam(list(model1.parameters()) + list(model2.parameters())
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

        for i in range(10000):

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

                    # print('1 x', x.shape)
                    x = embeddings[language](x).double() # batch_size x seq_len x emb_size
                    # print('1 x emb', x.shape)
                    shared_features = FS(x) # batch_size x seq_len x sf_size
                    # print('1 shared_features', shared_features.shape)
                    true_l = torch.Tensor([language_index]).expand(len(x), 1).long()
                    predicted_l = D(shared_features)
                    # print('1 Discriminator pred', predicted_l.shape)
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

                # print(' 2 x', x.shape)
                x = embeddings[language](x).double() # batch_size x seq_len x emb_size
                # print(' 2 x emb', x.shape)
                shared_features = FS(x) # batch_size x seq_len x sf_size
                # print(' 2 shared_features', shared_features.shape)
                gates, privates_features = FP(x)
                true_l = torch.Tensor([language_index]).expand(len(x)*self.seq_len, 1).long()
                gates = gates.view(gates.shape[0]*gates.shape[1], gates.shape[2])

                ypred = C(shared_features, privates_features) # batch_size x nout

                # print(' 2 classifier pred', ypred.shape)
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

                # print(' 3 x', x.shape)
                x = embeddings[language](x).double()
                # print(' 3 x emb', x.shape)
                shared_features = FS(x)
                # print(' 3 shared_features', shared_features.shape)
                predicted_l = D(shared_features)
                # print(' 3 Discriminator pred', predicted_l.shape)
                true_l = torch.Tensor([language_index]).expand(len(x), 1).long()
                total_loss += -lambda_1*D_loss(predicted_l, true_l.view(-1))

            total_loss.backward()
            # FP_optim.step()
            # FS_optim.step()
            # C_optim.step()
            c_optim.step()

            if i%10==0:
                try:
                    x_t, y_t = next(test_iter)
                except StopIteration:
                    test_iter = iter(test)
                    x_t, y_t = next(test_iter)
                with torch.no_grad():
                    x_t = embeddings[self.target_language](x_t).double()
                    _, fp_t = FP(x_t)
                    pred_t = C(FS(x_t), fp_t)
                    loss_test = float(C_loss(pred_t, y_t.view(-1)))
                    acc_test =  accuracy(pred_t.max(1)[1], y_t.view(-1))
                    test_plot.update(loss_test, acc_test)
                    print('Test loss = %.3f. Test accuracy = %.3f'%(loss_test, acc_test))



        train_plot.update(float(c_loss), float(c_accuracy), save=True)
        test_plot.update(float(loss_test), float(acc_test), save=True)


# todo mentor toutes les loss
#


if __name__ == '__main__':

    source_languages = ['fr'] #, 'de', 'jp'
    target_language = 'de'


    loader = AmazonReviewLoader(batch_size=16,
                                source_languages=source_languages,
                                target_language=target_language,
                                xml_limit_size=1000)

    labeled_corpus, unlabeled_corpus, test, embeddings_dicts = loader.load_corpus()

    model = Model(source_languages=source_languages,
                  target_language=target_language)


    model.train(labeled_corpus, unlabeled_corpus, test, embeddings_dicts)
