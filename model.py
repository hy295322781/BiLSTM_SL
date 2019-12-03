# -*- coding:utf-8 -*-
"""
Created on 2018-11-07 21:05:09

Author: Xiong Zecheng (295322781@qq.com)
"""
import random, time
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood, viterbi_decode
from data import pad_sequences, extract_specific_slot, evaluate


class BiLSTM_CRF(object):
    def __init__(self, embeddings, labels2idx, words2idx, intents2idx, config):
        self.embeddings = embeddings
        self.labels2idx = labels2idx
        self.num_labels = len(labels2idx)
        self.intents2idx = intents2idx
        self.num_intents = len(intents2idx)
        self.words2idx = words2idx
        self.config = config
        self.CRF = True
        self.shuffle = True
        self.early_termination = True
        self.slot_filling = True
        self.intent_detection = False
        self.slot_distinct = False
        self.hidden_dim = 300
        self.batch_size = 128
        self.epoch_num = 40
        self.termination_threshold = 3
        self.learning_rate = 0.001
        self.dropout_p = 0.9
        self.summary_path = "graph/."
        self.model_path = "checkpoint/variables"
        self.error_example_output = "output/error_example.txt"
        self.true_example_output = "output/true_example.txt"
        self.optimizer = "Adam"
        self.idx2words = dict()
        self.idx2labels = dict()
        for key, value in self.words2idx.items():
            self.idx2words[value] = key
        for key, value in self.labels2idx.items():
            self.idx2labels[value] = key

    def build_graph(self):
        tf.reset_default_graph()
        self.add_placeholders()
        self.lookup_layer_op()
        self.biLSTM_layer_op()
        self.softmax_pred_op()
        self.loss_op()
        self.trainstep_op()
        self.init_op()

    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
                                       name="word_ids")  # shape:[batchsize,maxtime] 长度不够补0
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")  # shape:[batchsize,maxtime]
        self.intents = tf.placeholder(tf.int32, shape=[None], name="intents")  # shape:[batchsize]
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")  # shape:[batchsize]

    def lookup_layer_op(self):
        with tf.variable_scope("words"):
            all_word_embeddings = tf.get_variable("all_word_embeddings",
                                                  dtype=tf.float32,
                                                  initializer=self.embeddings,
                                                  trainable=True)
            word_embeddings = tf.nn.embedding_lookup(params=all_word_embeddings,
                                                     ids=self.word_ids,
                                                     name="word_embeddings")
            self.word_embeddings = tf.nn.dropout(word_embeddings,
                                                 self.dropout_p)  # shape:[batchsize,maxtime,embedding_dim]

    def biLSTM_layer_op(self):
        with tf.variable_scope("bi-lstm"):
            cell_fw = LSTMCell(self.hidden_dim)
            cell_bw = LSTMCell(self.hidden_dim)
            # shape(singletrack):[batchsize,maxtime,hidden_dim] [batchsize,hidden_dim]
            (output_fw_seq, output_bw_seq), (output_fw_state, output_bw_state) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.word_embeddings,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)

        if self.slot_filling:
            with tf.variable_scope("slot_proj"):
                output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
                output = tf.nn.dropout(output, self.dropout_p)  # shape:[batchsize,maxtime,2*hidden_dim]
                W_s = tf.get_variable(name="W",
                                      shape=[2 * self.hidden_dim, self.num_labels],
                                      initializer=tf.contrib.layers.xavier_initializer(),
                                      dtype=tf.float32)
                b_s = tf.get_variable(name="b",
                                      shape=[self.num_labels],
                                      initializer=tf.zeros_initializer(),
                                      dtype=tf.float32)
                s = tf.shape(output)
                output = tf.reshape(output, [-1, 2 * self.hidden_dim])  # shape:[batchsize*maxtime,2*hidden_dim]
                pred_slot = tf.matmul(output, W_s) + b_s  # shape:[batchsize*maxtime,num_labels]
                self.logits_slot = tf.reshape(pred_slot,
                                              [-1, s[1], self.num_labels])  # shape:[batchsize,maxtime,num_labels]

        if self.intent_detection:
            with tf.variable_scope("intent_proj"):
                state = tf.concat([output_fw_state.h, output_bw_state.h], axis=-1)
                state = tf.nn.dropout(state, self.dropout_p)  # shape:[batchsize,hidden_dim*2]
                W_i = tf.get_variable(name="W",
                                      shape=[2 * self.hidden_dim, self.num_intents],
                                      initializer=tf.contrib.layers.xavier_initializer(),
                                      dtype=tf.float32)
                b_i = tf.get_variable(name="b",
                                      shape=[self.num_intents],
                                      initializer=tf.zeros_initializer(),
                                      dtype=tf.float32)
                self.logits_intent = tf.matmul(state, W_i) + b_i  # shape:[batchsize,num_intents]

    def softmax_pred_op(self):
        if self.slot_filling and not self.CRF:
            self.labels_softmax = tf.argmax(self.logits_slot, axis=-1)
            self.labels_softmax = tf.cast(self.labels_softmax, tf.int32)  # shape:[batchsize,maxtime]
        if self.intent_detection:
            self.intents_softmax = tf.argmax(self.logits_intent, axis=-1)
            self.intents_softmax = tf.cast(self.intents_softmax, tf.int32)  # shape:[batchsize]

    def loss_op(self):
        loss_slot = 0
        loss_intent = 0
        if self.slot_filling:
            if self.CRF:
                with tf.variable_scope("CRF"):
                    # log_likelihood shape:[batchsize], transition_params is a [num_labels,num_labels] transition matrix
                    log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits_slot,
                                                                                tag_indices=self.labels,
                                                                                sequence_lengths=self.sequence_lengths)
                    loss_slot = -tf.reduce_mean(log_likelihood)
                    self.loss = loss_slot
            else:
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_slot,
                                                                        # shape:[batchsize,maxtime]
                                                                        labels=self.labels)
                mask = tf.sequence_mask(self.sequence_lengths)
                losses = tf.boolean_mask(losses, mask)  # a vector only contain unmask elements
                loss_slot = tf.reduce_mean(losses)
                self.loss = loss_slot
        if self.intent_detection:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_intent,  # shape:[batchsize]
                                                                    labels=self.intents)
            loss_intent = tf.reduce_mean(losses)
            self.loss = loss_intent
        if self.slot_filling and self.intent_detection:
            self.loss = loss_slot + loss_intent

    def trainstep_op(self):
        if self.optimizer == 'Adam':
            optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer == 'Adadelta':
            optim = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer == 'Adagrad':
            optim = tf.train.AdagradOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer == 'RMSProp':
            optim = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer == 'Momentum':
            optim = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9)
        elif self.optimizer == 'SGD':
            optim = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        else:
            optim = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        grads_and_vars = optim.compute_gradients(self.loss)
        grads_and_vars_clip = [[tf.clip_by_value(g, -5.0, 5.0), v] for g, v in grads_and_vars]
        self.train_op = optim.apply_gradients(grads_and_vars_clip)

    def init_op(self):
        self.init_op = tf.global_variables_initializer()

    def train(self, train_set):
        """
        :param train:
        :return:
        """
        start = time.time()
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session(config=self.config) as sess:
            tf.get_default_graph().finalize()
            sess.run(self.init_op)
            self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)
            non_descent_step = 0
            min_loss = 10000
            for epoch in range(self.epoch_num):
                loss = self.run_one_epoch(sess, train_set, saver)
                print("epoch {}: loss {}".format(epoch+1, loss))
                if loss < min_loss:
                    min_loss = loss
                    non_descent_step = 0
                else:
                    non_descent_step += 1
                if non_descent_step >= self.termination_threshold and self.early_termination:
                    break
            print("train succeed!")
        end = time.time()
        print("process used:", end - start, "s.")

    def run_one_epoch(self, sess, train_set, saver):
        batch_num = (len(train_set[0]) - 1) // self.batch_size + 1
        if self.shuffle:
            zipped = list(zip(train_set[0], train_set[1], train_set[2]))
            random.shuffle(zipped)
            unzip = list(zip(*zipped))
            train_xs = unzip[0]
            train_y1s = unzip[1]
            train_y2s = unzip[2]
        else:
            train_xs = train_set[0]
            train_y1s = train_set[1]
            train_y2s = train_set[2]
        sum_loss = 0
        for batch in range(batch_num):
            if batch != batch_num - 1:
                sentences = train_xs[batch * self.batch_size:batch * self.batch_size + self.batch_size]
                labels = train_y1s[batch * self.batch_size:batch * self.batch_size + self.batch_size]
                intents = train_y2s[batch * self.batch_size:batch * self.batch_size + self.batch_size]
            else:
                sentences = train_xs[batch * self.batch_size:]
                labels = train_y1s[batch * self.batch_size:]
                intents = train_y2s[batch * self.batch_size:]
            feed_dict, _ = self.get_feed_dict(sentences, labels, intents)
            sess.run(self.train_op, feed_dict=feed_dict)
            sum_loss += sess.run(self.loss, feed_dict=feed_dict)
        saver.save(sess, self.model_path)
        return sum_loss / batch_num

    def get_feed_dict(self, seqs, labels=None, intents=None):
        word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)
        feed_dict = {self.word_ids: word_ids,
                     self.sequence_lengths: seq_len_list}
        if labels is not None:
            labels_, _ = pad_sequences(labels, pad_mark=0)
            feed_dict[self.labels] = labels_
        if intents is not None:
            feed_dict[self.intents] = intents
        return feed_dict, seq_len_list

    def test(self, test_set):
        real_seq = list()
        for seq in test_set[0]:
            real_seq.append(list(map(lambda x: self.idx2words[x], seq)))
        real_label = list()
        for seq in test_set[1]:
            real_label.append(list(map(lambda x: self.idx2labels[x], seq)))
        real_test_set = (real_seq, real_label)

        slot_predict = None
        intent_predict = None
        saver = tf.train.Saver()
        with tf.Session(config=self.config) as sess:
            saver.restore(sess, self.model_path)
            feed_dict, seq_len_list = self.get_feed_dict(test_set[0])
            if self.slot_filling:
                if self.CRF:
                    logits, transition_params = sess.run([self.logits_slot, self.transition_params],
                                                         feed_dict=feed_dict)
                    slot_predicts = list()
                    for logit, seq_len in zip(logits, seq_len_list):
                        viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                        slot_predicts.append(viterbi_seq)
                else:
                    slot_predicts = sess.run(self.labels_softmax, feed_dict=feed_dict)
                slot_predict = list()
                for i in range(len(test_set[0])):
                    seq_len = len(test_set[0][i])
                    predicted_seq = list(map(lambda x: self.idx2labels[x], slot_predicts[i][:seq_len]))
                    slot_predict.append(predicted_seq)
            if self.intent_detection:
                intent_predicts = sess.run(self.intents_softmax, feed_dict=feed_dict)
                for i in range(len(test_set[0])):
                    if test_set[2][i] == intent_predicts[i]:
                        # TODO intent eval
                        pass

        evaluate(real_test_set, slot_predict, intent_predict,
                 self.error_example_output, self.true_example_output, self.slot_distinct)

    def specific_test(self, test_set):
        slot_set = set()
        for key in self.labels2idx.keys():
            if key.startswith("B"):
                slot_set.add(key[2:])
        test_set_size = len(test_set[0])

        saver = tf.train.Saver()
        with tf.Session(config=self.config) as sess:
            saver.restore(sess, self.model_path)
            feed_dict, seq_len_list = self.get_feed_dict(test_set[0])
            if self.slot_filling:
                result_dict = dict()
                fr = open("analyze_addcrf/result.txt", "w")

                if self.CRF:
                    logits, transition_params = sess.run([self.logits_slot, self.transition_params],
                                                         feed_dict=feed_dict)
                    slot_predicts = list()
                    for logit, seq_len in zip(logits, seq_len_list):
                        viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                        slot_predicts.append(viterbi_seq)
                else:
                    slot_predicts = sess.run(self.labels_softmax, feed_dict=feed_dict)

                for slot in slot_set:
                    predicted_slot_count = 0
                    actual_slot_count = 0
                    hit_count = 0

                    fs = open("analyze_addcrf/" + slot + "_errcase.txt", "w")
                    for i in range(test_set_size):
                        sentence = list(map(lambda x: self.idx2words[x], test_set[0][i]))
                        seq_len = len(sentence)
                        predicted_seq = list(map(lambda x: self.idx2labels[x], slot_predicts[i][:seq_len]))
                        predict_specific_slot = extract_specific_slot(predicted_seq, slot)
                        label_seq = list(map(lambda x: self.idx2labels[x], test_set[1][i]))
                        actual_specific_slot = extract_specific_slot(label_seq, slot)

                        predict_specific_slot.sort()
                        actual_specific_slot.sort()

                        if predict_specific_slot != actual_specific_slot:
                            fs.write(str(i) + "\n")
                            for s in sentence:
                                fs.write(s + " ")
                            fs.write("\n")
                            for s in label_seq:
                                fs.write(s + " ")
                            fs.write("\n")
                            for s in predicted_seq:
                                fs.write(s + " ")
                            fs.write("\n")

                        for item in predict_specific_slot:
                            if item in actual_specific_slot:
                                hit_count += 1
                        predicted_slot_count += len(predict_specific_slot)
                        actual_slot_count += len(actual_specific_slot)

                        evaluation_dict = dict()
                        if predicted_slot_count != 0:
                            evaluation_dict["Precision"] = hit_count / predicted_slot_count
                        else:
                            evaluation_dict["Precision"] = "none predicted slot"
                        if actual_slot_count != 0:
                            evaluation_dict["Recall"] = hit_count / actual_slot_count
                        else:
                            evaluation_dict["Recall"] = "none actual slot"
                        if actual_slot_count + predicted_slot_count != 0:
                            evaluation_dict["F1score"] = 2 * hit_count / (actual_slot_count + predicted_slot_count)
                        else:
                            evaluation_dict["F1score"] = "none"
                        result_dict[slot] = evaluation_dict
                    fs.close()

                for k, v in result_dict.items():
                    fr.write(k + "\n")
                    for key, value in v.items():
                        fr.write(key + " " + str(value) + "\n")
                    fr.write("\n")
                fr.close()
