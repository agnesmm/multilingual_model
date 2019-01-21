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

class NeuralNetwork(Trainer):
    def __init__(self, n_epochs=10, language='fr'):
        self.data_loader = AmazonReviewLoader(language=language, multilingual_embeddings=False)
        self.embeddings = self.load_embedings()
        self.nout=1
        self.emb_shape = 300
        self.max_len = 200
        self.n_epochs = n_epochs

    def load_embedings(self):
        embeddings = self.data_loader.dictionary.embed
        embeddings = torch.Tensor(embeddings).type(torch.float)
        return nn.Embedding.from_pretrained(embeddings)

    def load_data(self):
        train_loader = self.data_loader.get_dataset('train', batch_size=32)
        test_loader = self.data_loader.get_dataset('test', batch_size=32)
        return train_loader, test_loader


    def train_model(self, train_loader, test_loader, model, loss = nn.MSELoss(),
                    optimizer=optim.SGD, optim_param={}, model_name='CNN'):

        train_plot = AccLossPlot(num=1, title='%s train'%model_name)
        test_plot = AccLossPlot(num=2, title='%s test'%model_name)

        print('train model')

        torch.manual_seed(SEED)
        random.seed(SEED)

        train_loss, val_loss, train_acc, val_acc = [], [1, 1], [], [1, 1]

        self.model = model(self.max_len, self.nout)
        self.loss = loss
        self.opt = optimizer(self.model.parameters(), **optim_param)

        for epoch_n in range(self.n_epochs):
            for batch_idx, (x, y) in enumerate(train_loader):

                x = self.embeddings(x)

                loss_t, acc_t = self.train(x.double(), y.double())
                train_loss.append(loss_t)
                train_acc.append(acc_t)

                if batch_idx%10==0:
                    print('Batch %d : loss = %.3f and accuracy = %.3f'%(batch_idx, loss_t, acc_t))


                train_plot.update(loss_t, acc_t)


            epoch_val_loss, epoch_val_acc = [], []
            for val_batch_idx, (x_val, y_val) in enumerate(test_loader):
                with torch.no_grad():
                    x_val = self.embeddings(x_val)
                    loss_v, acc_v = self.validation(x_val.double(), y_val.double())
                    epoch_val_loss.append(loss_v)
                    epoch_val_acc.append(acc_v)

                test_plot.update(loss_v, acc_v)

            val_loss.append(np.mean(epoch_val_loss))
            val_acc.append(np.mean(epoch_val_acc))
            print('Epoch %d. Val loss = %.3f. Val accuracy = %.3f'%(epoch_n,
                                                                    np.mean(epoch_val_loss),
                                                                    np.mean(epoch_val_acc)))

        train_plot.update(loss_t, acc_t, save=True)
        test_plot.update(loss_v, acc_v, save=True)

if __name__ == '__main__':
    baseline = NeuralNetwork(n_epochs=30)
    train_loader, test_loader = baseline.load_data()

    baseline.train_model(train_loader, test_loader,
                         ConvNetwork,
                         optimizer=optim.Adam,
                         optim_param={'lr' : 10e-3},
                         model_name='CNN')

    # baseline.train_model(train_loader, test_loader,
    #                      RNN,
    #                      optimizer=optim.Adam,
    #                      optim_param={'lr' : 10e-3})
