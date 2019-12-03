# -*- coding:utf-8 -*-
"""
Created on 2018-11-07 20:44:29

Author: Xiong Zecheng (295322781@qq.com)
"""
import numpy as np

def random_embedding(vocab, embedding_dim):
    """
    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat

def pad_sequences(sequences, pad_mark=0):
    """
    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x : len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list

def extract_slot(seq):
    slot = list()
    i = 0
    while i < len(seq):
        if seq[i].startswith("B"):
            slot_type = seq[i][2:]
            l = 1
            while i + l < len(seq) and seq[i + l] == "I-" + slot_type:
                l += 1
            slot.append((i, slot_type, l))  # (槽的位置,槽的类型,槽的长度)
            i = i + l
        else:
            i += 1
    return slot

def convert_slot(tuple_slot,tokens):
    '''
    将三元组的槽转换为键值对槽
    '''
    pair_slot = list()
    for i,slot_type,l in tuple_slot:
        slot_value = "".join(tokens[i:i+l])
        pair = (slot_type,slot_value)
        if pair not in pair_slot:
            pair_slot.append(pair)
    return pair_slot

def extract_specific_slot(seq,sp_slot):
    slot = list()
    i = 0
    while i < len(seq):
        if seq[i].startswith("B") and seq[i][2:]==sp_slot:
            l = 1
            while i + l < len(seq) and seq[i + l] == "I-" + sp_slot:
                l += 1
            slot.append((i, sp_slot, l))  # (槽的位置,槽的类型,槽的长度)
            i = i + l
        else:
            i += 1
    return slot

def evaluate(test_set,slot,intent,error_example_output,true_example_output,distinct=False):
    '''
    :param distinct: 是否将一个样本中的重复槽视为一个
    '''
    slot_hit_count = 0
    predict_slot_count = 0
    actual_slot_count = 0

    efw = open(error_example_output,'w')
    tfw = open(true_example_output,'w')
    for i,predict_seq in enumerate(slot):
        is_true = True
        predict_slot = extract_slot(predict_seq)
        actual_slot = extract_slot(test_set[1][i])
        if distinct:
            text_seq = test_set[0][i]
            predict_slot = convert_slot(predict_slot,text_seq)
            actual_slot = convert_slot(actual_slot,text_seq)
        for item in predict_slot:
            if item in actual_slot:
                slot_hit_count += 1
            else:
                is_true = False
        if not is_true:
            efw.write("{} actual_slot:{} predict_slot:{}\n".format(i+1,actual_slot,predict_slot))
            efw.write("{}\n".format(" ".join(test_set[0][i])))
            efw.write("{}\n".format(" ".join(test_set[1][i])))
            efw.write("{}\n\n".format(" ".join(predict_seq)))
        else:
            tfw.write("{} {} {}\n".format(i+1," ".join(test_set[0][i]),actual_slot))
        predict_slot_count += len(predict_slot)
        actual_slot_count += len(actual_slot)
    efw.close()
    print("test set size:{}".format(len(test_set[0])))
    print("predicted slot:{} actual slot:{} hit:{}".format(predict_slot_count,actual_slot_count,slot_hit_count))
    print("Precision:{}".format(slot_hit_count/predict_slot_count))
    print("Recall:{}".format(slot_hit_count/actual_slot_count))
    print("F1score:{}".format(2 * slot_hit_count / (actual_slot_count + predict_slot_count)))