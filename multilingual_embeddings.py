import fasttext
from fasttext import FastVector

import copy
from os import path

class MultiLingualEmbeddings(object):
    def __init__(self):
        self.embs_dir = '/Users/agnesmustar/0dac/multilingual_model/data/fastext/'
        # self.languages = ['fr', 'de', 'en', 'jp']
        self.languages = ['fr', 'de']
        self.matrix_dir = path.dirname(fasttext.__file__)


    def load_dictionary(self, language):
        vector_file = path.join(self.embs_dir + language, language + '.vec')
        dictionary = FastVector(vector_file=vector_file)
        return dictionary

    def project_dictionary(self, language, dictionary):
        multilingual_dictionary = copy.copy(dictionary)
        matrix_file = path.join(self.matrix_dir, 'alignment_matrices', language + '.txt')
        multilingual_dictionary.apply_transform(matrix_file)
        return multilingual_dictionary

    def load_dictionaries(self):
        dictionaries = {}
        multilingual_dictionaries = {}
        for l in self.languages:
            dictionaries[l] = self.load_dictionary(l)
            multilingual_dictionaries[l] = self.project_dictionary(l, dictionaries[l])
        return multilingual_dictionaries, dictionaries
