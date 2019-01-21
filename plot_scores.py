import matplotlib.pyplot as plt
plt.ion()
import numpy as np

# def accuracy(output, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)

#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, keep_all=False):
        self.reset()
        self.data = None
        if keep_all:
            self.data = []

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.data is not None:
            self.data.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class TrainLossPlot(object):
    def __init__(self):
        self.loss_train = []
        self.fig = plt.figure()

    def update(self, loss_train):
        self.loss_train.append(loss_train)

    def plot(self):
        plt.figure(self.fig.number)
        plt.clf()
        plt.plot(np.array(self.loss_train))
        plt.title("Train loss / batch")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.show()
        plt.draw_all()
        plt.pause(1e-3)


class AccLossPlot(object):
    def __init__(self, num=1, title=""):
        self.loss = []
        self.acc = []
        self.fig = plt.figure(figsize=(12,6), num=num)
        self.title = title

    def update(self, loss, acc, save=False):
        self.loss.append(loss)
        self.acc.append(acc)
        plt.figure(self.fig.number)
        plt.clf()

        plt.subplot(1,2,1)
        acc_array = np.array(self.acc)
        plt.plot(acc_array, label="acc")
        if len(acc_array) > 100:
            means = [np.mean(acc_array[i:100+i]) for i in range(len(acc_array)-100)]
            means = np.concatenate([np.array([np.nan]*99), means])
            plt.plot(means, label='average acc')

        plt.title("%s Accuracy / epoch"%self.title)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()


        plt.subplot(1,2,2)
        loss_array = np.array(self.loss)
        plt.plot(loss_array, label="loss")

        if len(loss_array) > 100:
            means = [np.mean(loss_array[i:100+i]) for i in range(len(loss_array)-100)]
            means = np.concatenate([np.array([np.nan]*99), means])
            plt.plot(means, label='average loss', color='red')

        plt.title("%s Loss / epoch"%self.title)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        plt.draw_all()
        plt.pause(1e-3)

        if save:
            plt.savefig('plot/%s'%self.title)


