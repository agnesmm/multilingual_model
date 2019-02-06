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


class SharedFeatureExtractorLSTM(nn.Module):
    def __init__(self, nin, nout=128, hidden_dim=64):
        super(SharedFeatureExtractorLSTM, self).__init__()
        # self.net = nn.Sequential(nn.Conv1d(nin, 1, kernel_size=2, stride=2, padding=0),
        #                          nn.MaxPool1d(2),
        #                          nn.Linear(75, nout),
        #                          nn.Dropout(p=0.3),
        #                          nn.Sigmoid())

        print('nin', nin)
        self.lstm = nn.LSTM(300, hidden_dim, num_layers=2,
                            bidirectional=False, dropout=.2)
        self.fc = nn.Linear(hidden_dim*2, nout)
        self.dropout = nn.Dropout(.2)
        self.a = nn.Sigmoid()


    def forward(self, x):
        x = x.double()
        x = x.permute(1, 0, 2)
        X = self.dropout(x)
        output, (hidden, cell) = self.lstm(x)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        y = self.fc(hidden.squeeze(0))
        y = nn.Sigmoid()(y)
        return y

class SharedFeatureExtractor(nn.Module):
    def __init__(self, emb_size, nout=128):
        super(SharedFeatureExtractor, self).__init__()
        self.hidden_size = 200
        # self.conv = nn.Conv2d(1, nout, kernel_size=3, padding=1)
        # self.maxpool = nn.MaxPool1d(kernel_size=2)
        # self.dropout = nn.Dropout(.2)
        self.fc1 = nn.Linear(emb_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, nout)

    # def forward(self, x):
    #     x = x.double().unsqueeze(1) # batch_size x 1 x seq_len x emb_size
    #     y = F.relu(self.conv(x)) # batch_size x fs_size x seq_len x emb_size
    #     print('y', y.shape)
    #     y = self.dropout(y)
    #     # y = y.permute(0,2,1,3) # batch_size x seq_len x fs_size x emb_size
    #     # y = [F.max_pool1d(i, 1).squeeze(1) for i in y]
    #     # y = torch.stack(y) # batch_size x seq_len x fs_size x emb_size/2
    #     y = F.relu(self.fc1(y))
    #     print('fc1', y.shape)
    #     y = F.relu(self.fc2(y)) # batch_size x seq_len x fs_size x 1
    #     print('fc2', y.shape)
    #     y = y.squeeze(3) # batch_size x fs_size x seq_len
    #     print('squeeze', y.shape)
    #     y = y.view(32, 200, 128) # batch_size x seq_len x fs_size
    #     print('view', y.shape)
    #     return y

    def forward(self, x):
        y = F.relu(self.fc1(x))
        y = nn.Dropout(.2)(y)
        y = nn.BatchNorm1d(self.hidden_size)(y)
        y = F.relu(self.fc2(y))
        y = nn.Dropout(.2)(y)
        y = nn.BatchNorm1d(self.hidden_size)(y)
        return y

class LanguageDiscriminator(nn.Module):
    def __init__(self, nin, nout, nhidden=64):
        super(LanguageDiscriminator, self).__init__()
        self.conv = nn.Conv1d(nin, 5, kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(5, nout)

    def forward(self, x):
        x = x.double() # batch_size x seq_len x fs_size
        y = self.conv(x) #batchsize x seq_len x ?
        y = F.max_pool1d(y, y.shape[2]) #batchsize x seq_len
        y = y.squeeze(2)
        y = self.dropout(y)
        y = self.fc(y) #batchsize x nout
        y = y.view(y.shape[0],-1)
        return y #batch_size x nout

# class Classifier(nn.Module):
#     def __init__(self, nin, nhidden=64):
#         super(Classifier, self).__init__()
#         self.net = nn.Sequential(nn.Linear(nin, nhidden),
#                                  nn.ReLU(),
#                                  nn.Linear(nhidden, 2),
#                                  nn.Sigmoid())

#     def forward(self, x):
#         return self.net(x)

# class Classifier(nn.Module):
#     def __init__(self, nin, nout=2, nhidden=64):
#         super(Classifier, self).__init__()
#         self.conv = nn.Conv1d(nin, 5, kernel_size=2, stride=2, padding=0)
#         self.dropout = nn.Dropout(p=0.3)
#         self.fc = nn.Linear(5, nout)

#     def forward(self, x):
#         x = x.double() # batch_size x seq_len x fs_size
#         y = self.conv(x) #batchsize x seq_len x ?
#         y = F.max_pool1d(y, y.shape[2]) #batchsize x seq_len
#         y = y.squeeze(2)
#         y = self.dropout(y)
#         y = self.fc(y) #batchsize x nout
#         y = y.view(y.shape[0],-1)
#         return y #batch_size x nout

class Classifier(nn.Module):
    def __init__(self, fs_size, nout):
        super(Classifier, self).__init__()
        self.hidden_size = 100
        self.fc1 = nn.Linear(200*fs_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, nout)

    def forward(self, x):
        # y = F.softmax(x, dim=1)
        # y = torch.sum(x, dim=1)
        # print('x', x.shape)
        y = x.view(-1, x.shape[1]*x.shape[2])
        # print('y dim -1', y.shape)
        # print('y softmax', y.shape)
        y = F.relu(self.fc1(y))
        # print('y fc1', y.shape)
        y = nn.Dropout(.2)(y)
        # print('y Dropout', y.shape)
        y = nn.BatchNorm1d(self.hidden_size)(y)
        # print('y BatchNorm1d', y.shape)
        y = F.relu(self.fc2(y))
        # print('y fc2', y.shape)
        y = nn.Dropout(.2)(y)
        # print('y Dropout', y.shape)
        y = nn.BatchNorm1d(self.hidden_size)(y)
        # print('y BatchNorm1d', y.shape)
        y = nn.LogSoftmax(dim=-1)(self.fc3(y))
        # print('y LogSoftmax', y.shape)
        return y

class Model(Trainer):
    def __init__(self, source_languages, target_language, d_iter=1):
        self.nout = 2
        self.emb_size = 300
        self.fs_size = 128
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


        emb_size = self.emb_size
        fs_size = self.fs_size
        seq_len = self.seq_len
        nout = self.nout
        languages = self.source_languages + [self.target_language]

        train_plot = AccLossPlot(num=1, title='shared features train')
        test_plot = AccLossPlot(num=2, title='shared features test')
        n_languages = len(list(labeled_corpus.keys()))


        FS = SharedFeatureExtractor(emb_size, fs_size)
        FS_optim = optim.Adam(FS.parameters())

        D = LanguageDiscriminator(seq_len, n_languages).double()
        D_loss = nn.NLLLoss()
        D_opt = optim.Adam(D.parameters())

        # FP = PrivateFeatureExtractor(self.seq_len, fs_size)

        C = Classifier(fs_size, nout)
        C_loss = nn.NLLLoss()
        C_optim = optim.Adam(C.parameters())


        def accuracy(ypred, ytrue):
            return float(ypred.eq(ytrue).sum())/float(len(ytrue))

        language_dict = {language:i for i, language in enumerate(labeled_corpus.keys())}
        labeled_iters = {l:iter(loader) for l, loader in labeled_corpus.items()}
        unlabeled_iters = {l:iter(loader) for l, loader in unlabeled_corpus.items()}
        embeddings = {l: self.get_embedings(embeddings_dicts[l]) for l in languages}

        for _ in range(1000):

            ###### train D ######
            d_accuracy = []
            for _ in range(self.d_iter):
                total_loss_d = 0
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
                    shared_features = FS(x) # batch_size x seq_len x fs_size
                    # print('1 shared_features', shared_features.shape)
                    true_l = torch.Tensor([language_index]).expand(len(x), 1).long()
                    D_opt.zero_grad()
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
                shared_features = FS(x) # batch_size x seq_len x fs_size
                # print(' 2 shared_features', shared_features.shape)
                ypred = C(shared_features) # batch_size x nout
                # print(' 2 classifier pred', ypred.shape)
                c_loss = C_loss(ypred, y.view(-1))
                c_accuracy = accuracy(ypred.max(1)[1], y.view(-1))
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

                # print(' 3 x', x.shape)
                x = embeddings[language](x).double()
                # print(' 3 x emb', x.shape)
                shared_features = FS(x)
                # print(' 3 shared_features', shared_features.shape)
                predicted_l = D(shared_features)
                # print(' 3 Discriminator pred', predicted_l.shape)
                true_l = torch.Tensor([language_index]).expand(len(x), 1).long()
                d_loss = -D_loss(predicted_l, true_l.view(-1))
                total_loss += 0.002*d_loss
            total_loss.backward()
            print('total_loss', total_loss)
            FS_optim.step()
            C_optim.step()

            for _, (x_t, y_t) in enumerate(test):
                with torch.no_grad():
                    x_t = embeddings[self.target_language](x_t)
                    pred_t = C(FS(x))
                    loss_test = float(C_loss(pred_t, y_t.view(-1)))
                    acc_test =  accuracy(pred_t.max(1)[1], y_t.view(-1))

                test_plot.update(loss_test, acc_test)
                print('Test loss = %.3f. Test accuracy = %.3f'%(loss_test, acc_test))

        train_plot.update(float(c_loss), float(c_accuracy), save=True)
        test_plot.update(float(loss_test), float(acc_test), save=True)


# todo mentor toutes les loss
#


if __name__ == '__main__':

    source_languages = ['fr', 'jp'] #, 'de', 'jp'
    target_language = 'de'


    loader = AmazonReviewLoader(batch_size=32,
                                source_languages=source_languages,
                                target_language=target_language)

    labeled_corpus, unlabeled_corpus, test, embeddings_dicts = loader.load_corpus()

    model = Model(source_languages=source_languages,
                  target_language=target_language)


    model.train(labeled_corpus, unlabeled_corpus, test, embeddings_dicts)
