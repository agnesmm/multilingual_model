#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
from os import path
import re
import string
import unicodedata

from pathos.multiprocessing import ProcessingPool as Pool
import nltk
import numpy as np
import os
#nltk.download('punkt')
import torch
import torch.utils.data
# import xml.etree.ElementTree as ET
import xml.etree.cElementTree as ET

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


class AmazonReviewLoaderByLanguage(dataLoader):
    def __init__(self, language = 'fr', datadir='data/cls-acl10-unprocessed',
                 multilingual_embeddings=True):
        super(AmazonReviewLoaderByLanguage, self).__init__()
        self.datadir = datadir
        self.language = language
        self.review_max_len = 200
        self.embeddings_dict = self.load_embeddings_dict(multilingual_embeddings)

    def load_embeddings_dict(self, multilingual_embeddings):
        mll = MultiLingualEmbeddings()
        embeddings_dict = mll.load_embeddings_dict(self.language)
        if multilingual_embeddings:
            mll.project_dictionary(self.language, embeddings_dict)
        return embeddings_dict

    def read_xml_file(self, part='train', domain='books', limit=10): #todo limit
        filename = path.join(self.datadir, self.language, domain, part+'.review')
        reviews, y = [], []

        root = ET.parse(filename).getroot()

        # todo pour aller + vite
        # if elem.tag == "record":
        #     ... process record element ...
        #     elem.clear()

        for i, items in enumerate(root):
            if limit is not None and i > limit:
                break
            if items.find('text').text is None:
                continue
            review = self.process_review(items.find('text').text)
            reviews.append(review)
            is_positive = self.process_ratings(items.find('rating').text)
            y.append(is_positive)
        return reviews, y

    # not faster with Pool()...
    # def get_Xy(self, items):
    #     if items.find('text').text is None:
    #         return None, None
    #     review = self.process_review(items.find('text').text)
    #     is_positive = self.process_ratings(items.find('rating').text)
    #     return review, is_positive

    # def read_xml_file(self, part='train', domain='books'):
    #     filename = path.join(self.datadir, self.language, domain, part+'.review')
    #     root = ET.parse(filename).getroot()
    #     pool = Pool(4)
    #     Xy = pool.map(self.get_Xy, root)
    #     reviews, y = zip(*Xy)
    #     return reviews, y

    def process_review(self, review):
        review = self.convert_to_string(review)
        review = self.remove_ponctuation(str(review))
        review = review.lower()
        review = self.remove_numerical(review)
        review = nltk.word_tokenize(review)
        return review

    def process_ratings(self, y):
        return int(float(y)>=3)

    def encode_review(self, review):
        review_max_len = self.review_max_len
        review = [self.embeddings_dict.word2id.get(token, 1) for token in review][:review_max_len]
        if len(review) < review_max_len:
            review.extend([0 for _ in range(review_max_len - len(review))])
        return review

    def encode_reviews(self, reviews):
        reviews = np.array([self.encode_review(review) for review in reviews if review is not None])
        return torch.Tensor(reviews).type(torch.long)

    @timeit
    def get_dataloader(self, batch_size, part='train'):
        reviews, y = self.read_xml_file(part=part)
        X = self.encode_reviews(reviews)
        y = torch.Tensor([y_i for y_i in y if y_i is not None]).type(torch.long)
        y = y.unsqueeze(1)
        dataset = torch.utils.data.TensorDataset(X, y)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size)

class AmazonReviewLoader(object):
    def __init__(self, source_languages, target_language, batch_size=32):
        self.batch_size = batch_size
        self.source_languages = source_languages
        self.target_language = target_language
        self.languages = source_languages + [target_language]

    @timeit
    def load_corpus(self):
        labeled_corpus, unlabeled_corpus, embeddings_dicts = {}, {}, {}
        for l in self.languages:
            print('LOADING LANGUAGE %s'%(l))
            dataloader = AmazonReviewLoaderByLanguage(language=l)
            labeled_corpus[l] = dataloader.get_dataloader(part='train', batch_size=self.batch_size)
            # unlabeled_corpus[l] = dataloader.get_dataloader(part='unlabeled', batch_size=self.batch_size)
            embeddings_dicts[l] = dataloader.embeddings_dict
            if l==self.target_language:
                test = dataloader.get_dataloader(part='test', batch_size=self.batch_size)
        return labeled_corpus, unlabeled_corpus, test, embeddings_dicts



if __name__ == '__main__':

    source_languages = ['fr'] #, 'de', 'jp'
    target_language = 'de'

    loader = AmazonReviewLoader(batch_size=32,
                                source_languages=source_languages,
                                target_language=target_language)

    labeled_corpus, unlabeled_corpus, test, embeddings_dicts = loader.load_corpus()






