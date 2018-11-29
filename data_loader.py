import abc
from os import path

import xml.etree.ElementTree as ET





class dataLoader(object):
    def __init__(self):
        pass

    @abc.abstractmethod
    def load_data(self):
        raise NotImplementedError


class AmazonReviewLoader(dataLoader):
    def __init__(self):
        super(AmazonReviewLoader, self).__init__()
        self.datadir = 'data/cls-acl10-unprocessed'
        self.languages = ['de', 'fr', 'en', 'jp']

    def load_dataset(self, filename):
        reviews, y = [], []
        root = ET.parse(filename).getroot()
        for items in root:
            reviews.append(items.find('rating').text)
            y.append(items.find('text').text)
        return reviews, y


    def get_dataset(self):
        reviews, y, languages = [], [], []
        for language in self.languages:
            filename = path.join(self.datadir, language, 'books/train.review')
            reviews_l, y_l = self.load_dataset(filename)
            reviews.extend(reviews_l)
            y.extend(y_l)
            languages.extend([language]*len(y_l))

        print(len(reviews), len(y), len(languages))



if __name__ == '__main__':


    data_loader = AmazonReviewLoader()
    dataset = data_loader.get_dataset()

