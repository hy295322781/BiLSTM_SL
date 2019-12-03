# -*- coding:utf-8 -*-
"""
Created on 2019-01-23 09:16:09

Author: Xiong Zecheng (295322781@qq.com)
"""
import pickle

def read_record(path):
    with open(path,"r") as f:
        sentences = list()
        labels = list()
        intents = list()

        for line in f.readlines():
            pair = line.strip().split("\t")

            sentence = pair[0].split(" ")[1:-1]
            for i in range(len(sentence)):
                if sentence[i].isdigit():
                    sentence[i] = "DIGIT"
            sentences.append(sentence)

            label = pair[1].split(" ")[1:-1]
            labels.append(label)

            intent = pair[1].split(" ")[-1]
            intents.append(intent)

            if len(sentence)!=len(label):
                print(line)
                print(sentence)
                print(label)

        return sentences,labels,intents

def read_dataset(path,word2idx,label2idx,intent2idx):
    sentences, labels, intents = read_record(path)
    data_set = list()
    indexed_sentences = list()
    indexed_labels = list()
    indexed_intents = list()

    for sentence in sentences:
        indexed_sentence = list()
        for word in sentence:
            if word not in word2idx.keys():
                word2idx[word] = len(word2idx)
            indexed_sentence.append(word2idx[word])
        indexed_sentences.append(indexed_sentence)

    for label in labels:
        indexed_label = list()
        for l in label:
            if l not in label2idx.keys():
                label2idx[l] = len(label2idx)
            indexed_label.append(label2idx[l])
        indexed_labels.append(indexed_label)

    for intent in intents:
        if intent not in intent2idx.keys():
            intent2idx[intent] = len(intent2idx)
        indexed_intents.append(intent2idx[intent])

    data_set.append(indexed_sentences)
    data_set.append(indexed_labels)
    data_set.append(indexed_intents)
    return data_set

def dump_atis_data():
    word2idx = {"[PAD]": 0}
    label2idx = {"[PAD]": 0}
    intent2idx = {}

    train_set = read_dataset("data/atis.train.w-intent.iob", word2idx, label2idx, intent2idx)
    test_set = read_dataset("data/atis.test.w-intent.iob", word2idx, label2idx, intent2idx)

    d = dict()
    d["words2idx"] = word2idx
    d["labels2idx"] = label2idx
    d["intents2idx"] = intent2idx

    with open("data/atis.fold0.pkl", "wb") as f:
        pickle.dump((train_set, test_set, d), f)

if __name__=="__main__":
    dump_atis_data()