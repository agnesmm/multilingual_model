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
        self.conv = nn.Conv2d(1, nout, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(.2)
        self.fc1 = nn.Linear(300, 150)
        self.fc2 = nn.Linear(150, 1)

    def forward(self, x):
        x = x.double().unsqueeze(1) # batch_size x 1 x seq_len x emb_size
        y = F.relu(self.conv(x)) # batch_size x features_size x seq_len x emb_size
        y = self.dropout(y)
        # y = y.permute(0,2,1,3) # batch_size x seq_len x features_size x emb_size
        # y = [F.max_pool1d(i, 1).squeeze(1) for i in y]
        # y = torch.stack(y) # batch_size x seq_len x features_size x emb_size/2
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y)) # batch_size x seq_len x features_size x 1
        y = y.squeeze(3) # batch_size x features_size x seq_len
        y = y.view(32, 200, 128) # batch_size x seq_len x features_size
        return y


class LanguageDiscriminator(nn.Module):
    def __init__(self, nin, nout, nhidden=64):
        super(LanguageDiscriminator, self).__init__()
        self.conv = nn.Conv1d(nin, 5, kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(5, nout)

    def forward(self, x):
        x = x.double() # batch_size x seq_len x features_size
        y = self.conv(x) #batchsize x seq_len x ?
        y = F.max_pool1d(y, y.shape[2]) #batchsize x seq_len
        y = y.squeeze(2)
        y = self.dropout(y)
        y = self.fc(y) #batchsize x nout
        y = y.view(y.shape[0],-1)
        return y #batch_size x nout

# class TaskPredictor(nn.Module):
#     def __init__(self, nin, nhidden=64):
#         super(TaskPredictor, self).__init__()
#         self.net = nn.Sequential(nn.Linear(nin, nhidden),
#                                  nn.ReLU(),
#                                  nn.Linear(nhidden, 2),
#                                  nn.Sigmoid())

#     def forward(self, x):
#         return self.net(x)
class TaskPredictor(nn.Module):
    def __init__(self, nin, nout, nhidden=64):
        super(TaskPredictor, self).__init__()
        self.conv = nn.Conv1d(nin, 5, kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(5, nout)

    def forward(self, x):
        x = x.double() # batch_size x seq_len x features_size
        y = self.conv(x) #batchsize x seq_len x ?
        y = F.max_pool1d(y, y.shape[2]) #batchsize x seq_len
        y = y.squeeze(2)
        y = self.dropout(y)
        y = self.fc(y) #batchsize x nout
        y = y.view(y.shape[0],-1)
        return y #batch_size x nout



class ConvNetwork(nn.Module):
    def __init__(self, nin, nout):
        super(ConvNetwork, self).__init__()

        self.conv1 = nn.Conv1d(nin, 1, kernel_size=2, stride=2, padding=0)
        self.max1 = nn.MaxPool1d(2)
        self.fc = nn.Linear(75, nout, bias=True)
        self.dropout = nn.Dropout(p=0.3)
        self.a = nn.Sigmoid()


    def forward(self, x):
        y = self.conv1(x)
        y = self.max1(y)
        # print('y', y.shape)
        y=self.fc(y)
        # print('y', y.shape)
        y = self.dropout(y)
        y=self.a(y)
        # print('y', y.shape)
        y=y.view(y.shape[0],-1)
        # print('y', y.shape)
        return y



class Model(Trainer):
    def __init__(self, source_languages, target_language, d_iter=10):
        self.nout=1
        self.emb_shape = 300
        self.max_len = 200
        self.d_iter = d_iter
        self.source_languages = source_languages
        self.target_language = target_language


    def get_embedings(self, embeddings_dict):
        embeddings = embeddings_dict.embed
        embeddings = torch.Tensor(embeddings).type(torch.float)
        return nn.Embedding.from_pretrained(embeddings)

    def train(self, labeled_corpus, unlabeled_corpus, test, embeddings_dicts):

        features_size = 128
        emb_size = 300
        seq_len = 200

        train_plot = AccLossPlot(num=1, title='shared features train')
        n_languages = len(list(labeled_corpus.keys()))


        FS = SharedFeatureExtractor(self.max_len, features_size)
        FS_optim = optim.Adam(FS.parameters())

        D = LanguageDiscriminator(seq_len, n_languages).double()
        D_loss = nn.NLLLoss()
        D_opt = optim.Adam(D.parameters())

        # FP = PrivateFeatureExtractor(self.max_len, features_size)

        C = TaskPredictor(features_size)
        C_loss = nn.NLLLoss()
        C_optim = optim.Adam(C.parameters())


        def accuracy(ypred, ytrue):
            return float(ypred.eq(ytrue).sum())/float(len(ytrue))

        language_dict = {language:i for i, language in enumerate(labeled_corpus.keys())}
        iters = {l:iter(loader) for l, loader in labeled_corpus.items()}
        embeddings = {l: self.get_embedings(embeddings_dicts[l]) for l in iters.keys()}

        for _ in range(100):

            ###### train D ######
            for _ in range(self.d_iter):
                total_loss_d = 0
                for language, dataloader in labeled_corpus.items():
                    language_index = language_dict[language]
                    iter_l = iters[language]

                    # endless iter
                    try:
                        x, y = next(iter_l)
                    except StopIteration:
                        iter_l = iter(labeled_corpus[language])
                        iters[language] = iter_l
                        x, y = next(iter_l)

                    x = embeddings[language](x) # batch_size x seq_len x emb_size
                    shared_features = FS(x) # batch_size x seq_len x features_size
                    true_l = torch.Tensor([language_index]).expand(len(x), 1).long()
                    D_opt.zero_grad()
                    predicted_l = D(shared_features)
                    d_loss = D_loss(predicted_l, true_l.view(-1))
                    total_loss_d += d_loss
                total_loss_d.backward()
                D_opt.step()

            ###### train FS and C ######
            total_loss = 0
            for language, dataloader in labeled_corpus.items():
                language_index = language_dict[language]
                iter_l = iters[language]

                # endless iter
                try:
                    x, y = next(iter_l)
                except StopIteration:
                    iter_l = iter(labeled_corpus[language])
                    iters[language] = iter_l
                    x, y = next(iter_l)

                x = self.embeddings[language](x) # batch_size x seq_len x emb_size
                shared_features = FS(x) # batch_size x seq_len x features_size
                ypred = C(shared_features) # batch_size x nout
                c_loss = C_loss(ypred, y.view(-1))
                c_accuracy = accuracy(ypred.max(1)[1], y.view(-1))
                train_plot.update(c_loss, c_accuracy)
                total_loss += c_loss


            for language, dataloader in labeled_corpus.items():
                language_index = language_dict[language]
                iter_l = iters[language]

                # endless iter
                try:
                    x, y = next(iter_l)
                except StopIteration:
                    iter_l = iter(labeled_corpus[language])
                    iters[language] = iter_l
                    x, y = next(iter_l)

                x = self.embeddings[language](x)
                shared_features = FS(x)
                predicted_l = D(shared_features)
                true_l = torch.Tensor([language_index]).expand(len(x), 1).long()
                d_loss = -D_loss(predicted_l, true_l.view(-1))
                total_loss += 0.002*d_loss
            total_loss.backward()
            print('total_loss', total_loss)
            FS_optim.step()
            C_optim.step()

        train_plot.update(float(c_loss), c_accuracy, save=True)





if __name__ == '__main__':

    source_languages = ['fr'] #, 'de', 'jp'
    target_language = 'de'


    loader = AmazonReviewLoader(batch_size=32,
                                source_languages=source_languages,
                                target_language=target_language)

    labeled_corpus, unlabeled_corpus, test, embeddings_dicts = loader.load_corpus()

    model = Model(source_languages=source_languages,
                  target_language=target_language)


    model.train(labeled_corpus, unlabeled_corpus, test, embeddings_dicts)
