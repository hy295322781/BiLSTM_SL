# -*- coding:utf-8 -*-
"""
Created on 2018-11-07 17:39:54

Author: Xiong Zecheng (295322781@qq.com)
"""
import os,pickle
import tensorflow as tf
from data import random_embedding
from model import BiLSTM_CRF

class Processor():
    def __init__(self,path):
        f = open(path, 'rb')
        try:
            self.train_set, self.test_set, self.dicts = pickle.load(f, encoding='latin1')
        except:
            self.train_set, self.test_set, self.dicts = pickle.load(f)
        f.close()
        self.embeddings = random_embedding(self.dicts["words2idx"], 300)

    def conf(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"  # default: 0
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        return config

    def train(self):
        config = self.conf()
        model = BiLSTM_CRF(self.embeddings, self.dicts["labels2idx"], self.dicts["words2idx"],
                           self.dicts["intents2idx"], config=config)
        model.build_graph()
        model.train(self.train_set)

    def test(self):
        config = self.conf()
        model = BiLSTM_CRF(self.embeddings, self.dicts["labels2idx"], self.dicts["words2idx"],
                           self.dicts["intents2idx"], config=config)
        model.build_graph()
        model.test(self.test_set)

if __name__ == "__main__":
    atis = Processor("data/atis.fold0.pkl")
    atis.train()
    atis.test()