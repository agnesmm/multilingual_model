#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
from os import path
import re
import string
import unicodedata

import nltk
import numpy as np
#nltk.download('punkt')
import torch
import torch.utils.data
import xml.etree.ElementTree as ET

from multilingual_embeddings import MultiLingualEmbeddings
from utils import timeit


class dataLoader(object):
    def __init__(self):
        pass

    @abc.abstractmethod
    def load_data(self):
        raise NotImplementedError

    def remove_ponctuation(self, review):
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        return regex.sub('', review)

    def convert_to_string(self, review):
        if not type(review) is str:
            review = unicodedata.normalize('NFKD', review).encode('ascii','ignore')
        return review

    def remove_numerical(self, review):
        return re.sub('\d', '', review)


class AmazonReviewLoader(dataLoader):
    def __init__(self, language = 'fr', datadir='data/cls-acl10-unprocessed',
                 multilingual_embeddings=True):
        super(AmazonReviewLoader, self).__init__()
        self.datadir = datadir
        self.language = language
        self.review_max_len = 200
        self.dictionary = self.load_dict(multilingual_embeddings)

    def load_dict(self, multilingual_embeddings):
        mll = MultiLingualEmbeddings()
        dictionary = mll.load_dictionary(self.language)
        if multilingual_embeddings:
            mll.project_dictionary(self.language, dictionary)
        return dictionary

    def read_xml_file(self, part='train', domain='books'):
        filename = path.join(self.datadir, self.language, domain, part+'.review')
        reviews, y = [], []
        root = ET.parse(filename).getroot()
        for i, items in enumerate(root):
            review = self.process_review(items.find('text').text)
            reviews.append(review)
            is_positive = self.process_ratings(items.find('rating').text)
            y.append(is_positive)
        return reviews, y

    def process_review(self, review):
        review = self.convert_to_string(review)
        review = self.remove_ponctuation(str(review))
        review = review.lower()
        review = self.remove_numerical(review)
        review = nltk.word_tokenize(review)
        return review

    def process_ratings(self, y):
        y = float(y)
        if y<3:
            return 0
        else:
            return 1

    def encode_review(self, review):
        review_max_len = self.review_max_len
        review = [self.dictionary.word2id.get(token, 1) for token in review][:review_max_len]
        if len(review) < review_max_len:
            review.extend([0 for _ in xrange(review_max_len - len(review))])
        return review

    def encode_reviews(self, reviews):
        reviews = np.array([self.encode_review(review) for review in reviews])
        return torch.Tensor(reviews).type(torch.long)


    def get_dataset(self, part='train', batch_size=32):
        reviews, y = self.read_xml_file(part=part)
        X = self.encode_reviews(reviews)
        y = torch.Tensor(y).type(torch.long)
        y = y.unsqueeze(1)
        dataset = torch.utils.data.TensorDataset(X, y)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


if __name__ == '__main__':

    fr_data_loader = AmazonReviewLoader(language='fr', multilingual_embeddings=False)
    de_data_loader = AmazonReviewLoader(language='de', multilingual_embeddings=False)

    train_loader = fr_data_loader.get_dataset('train')
    test_loader = de_data_loader.get_dataset('train')

