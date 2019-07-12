# coding:utf-8

# -------------------------------------------------------------------------------
# @Author        chenfeiyu01
# @Name:         Model.py
# @Project       couplet
# @Product       PyCharm
# @DateTime:     2019-06-26 17:02
# @Contact       chenfeiyu01@baidu.com
# @Version       1.0
# @Description:
# -------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow.contrib import keras
from tensorflow.python.util import nest
from tensorflow.contrib.seq2seq import BahdanauAttention
from tensorflow.contrib.seq2seq import TrainingHelper
from tensorflow.contrib.seq2seq import dynamic_decode
from tensorflow.contrib.seq2seq import sequence_loss
from tensorflow.contrib.seq2seq import AttentionWrapper
from tensorflow.contrib.seq2seq import GreedyEmbeddingHelper
from tensorflow.contrib.seq2seq import BeamSearchDecoder


class Seq2Seq(object):
    def __init__(self, up_link,
                 encode_lengths,
                 down_link,
                 decode_lengths,
                 vocab_size,
                 hidden_size,
                 embedding_size,
                 dropout,
                 l2_regularizer,
                 base_learn_rate,
                 max_length,
                 start_token,
                 end_token,
                 beam_search,
                 beam_size,
                 layer_size):
        self.__encode_lengths = encode_lengths
        self.__up_link = up_link
        self.__decode_lengths = decode_lengths
        self.__down_link = down_link
        self.__vocab_size = vocab_size
        self.__hidden_size = hidden_size
        self.__embedding_size = embedding_size
        self.__dropout = dropout
        self.__base_learn_rate = base_learn_rate
        self.__l2_regularizer = l2_regularizer
        self.__max_length = max_length
        self.__start_token = start_token
        self.__end_token = end_token
        self.__beam_search = beam_search
        self.__beam_size = beam_size
        self.__layer_size = layer_size
        self.__addEmbeddingLayer()
        self.__addEncodingLayer()
        self.__reduce_states()

    def __addEmbeddingLayer(self):
        with tf.name_scope("embeddingLayer"):
            with tf.variable_scope('reduceWeights', reuse=tf.AUTO_REUSE):
                self.__embedding = tf.get_variable(name='embedding', dtype=tf.float32,
                                                   shape=[self.__vocab_size, self.__embedding_size],
                                                   initializer=tf.contrib.layers.xavier_initializer())
            embedding_up_link = tf.nn.embedding_lookup(self.__embedding, self.__up_link)
            self.__embedding_up_link = tf.nn.dropout(embedding_up_link, rate=1 - self.__dropout)

    def __addEncodingLayer(self):
        with tf.name_scope("encodingLayer"):
            layer_size = self.__layer_size // 2
            # fw_cell = tf.nn.rnn_cell.DropoutWrapper(cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=self.__hidden_size),
            #                                         output_keep_prob=self.__dropout)
            fw_cell = tf.nn.rnn_cell.MultiRNNCell(
                [tf.nn.rnn_cell.DropoutWrapper(cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=self.__hidden_size),
                                               output_keep_prob=self.__dropout) for _ in range(layer_size)])
            # bw_cell = tf.nn.rnn_cell.DropoutWrapper(cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=self.__hidden_size),
            #                                         output_keep_prob=self.__dropout)
            bw_cell = tf.nn.rnn_cell.MultiRNNCell(
                [tf.nn.rnn_cell.DropoutWrapper(cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=self.__hidden_size),
                                               output_keep_prob=self.__dropout) for _ in range(layer_size)])
            bid_output, (fw_st, bw_st) = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                         cell_bw=bw_cell,
                                                                         sequence_length=self.__encode_lengths,
                                                                         inputs=self.__embedding_up_link,
                                                                         dtype=tf.float32)
            self.__bid_output = tf.concat(bid_output, -1)
            fw_c = tf.concat([_.c for _ in fw_st], axis=-1)
            fw_h = tf.concat([_.h for _ in fw_st], axis=-1)
            self.__fw_st = tf.nn.rnn_cell.LSTMStateTuple(fw_c, fw_h)
            bw_c = tf.concat([_.c for _ in bw_st], axis=-1)
            bw_h = tf.concat([_.h for _ in bw_st], axis=-1)
            self.__bw_st = tf.nn.rnn_cell.LSTMStateTuple(bw_c, bw_h)

    def __reduce_states(self):
        with tf.name_scope("reduceStatesLayer"):
            old_c = tf.concat([self.__fw_st.c, self.__bw_st.c], axis=-1)
            old_h = tf.concat([self.__fw_st.h, self.__bw_st.h], axis=-1)
            with tf.variable_scope('reduceWeights', reuse=tf.AUTO_REUSE):
                weights = tf.get_variable(name='reduceWeights',
                                          shape=[self.__layer_size * self.__hidden_size, self.__hidden_size],
                                          initializer=tf.contrib.layers.xavier_initializer())
                bias = tf.get_variable(name='reduceBias', shape=(self.__hidden_size,),
                                       initializer=tf.contrib.layers.xavier_initializer())
            new_c = tf.nn.relu(tf.nn.xw_plus_b(old_c, weights, bias))
            new_h = tf.nn.relu(tf.nn.xw_plus_b(old_h, weights, bias))
            self.__decode_init_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)

    def __addDecodingLayer(self, mode):
        with tf.name_scope('decodingLayer'):
            batch_size = tf.shape(self.__up_link)[0]
            if self.__beam_search and mode == tf.estimator.ModeKeys.PREDICT:
                self.__bid_output = tf.contrib.seq2seq.tile_batch(self.__bid_output, multiplier=self.__beam_size)
                self.__decode_init_state = tf.contrib.seq2seq.tile_batch(self.__decode_init_state,
                                                                         multiplier=self.__beam_size)
                self.__encode_lengths = tf.contrib.seq2seq.tile_batch(self.__encode_lengths,
                                                                      multiplier=self.__beam_size)
                batch_size = batch_size * self.__beam_size
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=self.__hidden_size,
                                                                       memory=self.__bid_output,
                                                                       memory_sequence_length=self.__encode_lengths)
            decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.__hidden_size, dtype=tf.float32)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell,
                                                               attention_mechanism=attention_mechanism,
                                                               attention_layer_size=self.__hidden_size,
                                                               alignment_history=True)#记录每个attention值
            output_layer = tf.layers.Dense(units=self.__vocab_size,
                                           activation=tf.nn.relu,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           kernel_regularizer=keras.regularizers.l2(self.__l2_regularizer))
            initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
            initial_state = initial_state.clone(cell_state=self.__decode_init_state)
            if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
                embedding_down_link_input = tf.nn.embedding_lookup(self.__embedding, self.__down_link)
                training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=embedding_down_link_input,
                                                                    sequence_length=self.__decode_lengths,
                                                                    name='training_helper')
                training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                                   helper=training_helper,
                                                                   initial_state=initial_state,
                                                                   output_layer=output_layer)  # 调用dynamic_decode进行解码，decoder_outputs是一个namedtuple，里面包含两项(rnn_outputs, sample_id)
                max_length = tf.reduce_max(self.__decode_lengths)
                decoder_outputs, attention, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
                                                                          maximum_iterations=max_length,
                                                                          impute_finished=True)  # 遇到EOS自动停止解码（EOS之后的所有time step的输出为0，输出状态为最后一个有效time step的输出状态）
                self.attention = tf.transpose(attention.alignment_history.stack(), perm=[1, 0, 2])#得到每个attention值
                weights = tf.sequence_mask(self.__decode_lengths, dtype=tf.float32)
                down_link_output = tf.strided_slice(self.__down_link, begin=[0, 1], end=tf.shape(self.__down_link))
                self.loss = tf.contrib.seq2seq.sequence_loss(logits=decoder_outputs.rnn_output,
                                                             targets=down_link_output,
                                                             weights=weights)
                # self.logits = tf.where(tf.not_equal(decoder_outputs.rnn_output, 0.0))
                self.decode_input = self.__down_link
                self.decode_output = down_link_output
                # tf.summary.scalar('loss', self.loss)
                # self.summary_op = tf.summary.merge_all()
            else:
                start_tokens = tf.fill([tf.shape(self.__up_link)[0]], self.__start_token)
                if self.__beam_search:
                    inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell,
                                                                             embedding=self.__embedding,
                                                                             start_tokens=start_tokens,
                                                                             end_token=self.__end_token,
                                                                             initial_state=initial_state,
                                                                             beam_width=self.__beam_size,
                                                                             output_layer=output_layer)
                else:
                    decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.__embedding,
                                                                               start_tokens=start_tokens,
                                                                               end_token=self.__end_token)
                    inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                                        helper=decoding_helper,
                                                                        initial_state=initial_state,
                                                                        output_layer=output_layer)
                decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder,
                                                                          maximum_iterations=self.__max_length)
                if self.__beam_search:
                    self.decoder_predict_decode = {'up_link': self.__up_link,
                                                   'down_link': tf.transpose(decoder_outputs.predicted_ids,
                                                                             perm=[0, 2, 1])}
                else:
                    self.decoder_predict_decode = {'up_link': self.__up_link, 'down_link': decoder_outputs.sample_id}

    def getResult(self, mode):
        self.__addDecodingLayer(mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            learn_rate = tf.train.exponential_decay(self.__base_learn_rate,
                                                    tf.train.get_global_step(),
                                                    100,
                                                    0.98,
                                                    staircase=True)
            optimizer = tf.train.AdamOptimizer(0.001)
            gradients = optimizer.compute_gradients(self.loss)
            clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
            self.train_op = optimizer.apply_gradients(clipped_gradients, global_step=tf.train.get_global_step())
            return self.loss, self.train_op, self.decode_input, self.decode_output
        if mode == tf.estimator.ModeKeys.EVAL:
            return self.loss
        else:
            return self.decoder_predict_decode
