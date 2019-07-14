# coding:utf-8

# -------------------------------------------------------------------------------
# @Author        chenfeiyu01
# @Name:         Main.py
# @Project       couplet
# @Product       PyCharm
# @DateTime:     2019-06-26 17:02
# @Contact       chenfeiyu01@baidu.com
# @Version       1.0
# @Description:
# -------------------------------------------------------------------------------
from pathlib import Path
import tensorflow as tf
import functools
import argparse, os, json
import numpy as np
from dataHelper import DataHelper
from Model import Couplet
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)  # 实例化app对象

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# from tensorflow.contrib.estimator

# tf.enable_eager_execution()
parser = argparse.ArgumentParser(description='Seq2Seq超参数设置')
parser.add_argument('--dataPath', type=str, default='couplet', help='数据目录')
parser.add_argument('--hidden_size', type=int, default=512, help='隐藏层维度')
parser.add_argument('--embedding_size', type=int, default=128, help='词向量维度')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
parser.add_argument('--base_learnRate', type=float, default=0.001, help='初始学习率')
parser.add_argument('--l2_regularizer', type=float, default=0.001, help='l2正则项系数')
parser.add_argument('--batch_size', type=int, default=32, help='batchSize')
parser.add_argument('--num_epoch', type=int, default=50, help='epoches')
parser.add_argument('--max_length', type=int, default=100, help='序列最大长度')
parser.add_argument('--model_path', type=str, default='model', help='模型保存路径')
parser.add_argument('--beam_search', type=bool, default=True, help='是否使用BeamSearch')
parser.add_argument('--beam_size', type=bool, default=5, help='beam_size')
parser.add_argument('--encode_layer_size', type=bool, default=6, help='编码层bisltm层数')

FLAGS = parser.parse_args()


def model_fn(features, labels, mode, params):
    up_link, encode_lengths = features
    down_link, decode_lengths = None, None
    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
        down_link, decode_lengths = labels
    model = Couplet(up_link,
                    encode_lengths,
                    down_link,
                    decode_lengths,
                    params['vocabs'],
                    params['hiddenSize'],
                    params['embeddingSize'],
                    dropout=params['dropout'] if mode == tf.estimator.ModeKeys.TRAIN else 1.0,
                    l2_regularizer=params['l2Regularizer'],
                    base_learn_rate=params['baseLearnRate'],
                    max_length=params['maxLength'],
                    start_token=params['startToken'],
                    end_token=params['endToken'],
                    beam_search=params['beamSearch'],
                    beam_size=params['beamSize'],
                    layer_size=params['encodeLayerSize'])
    if mode == tf.estimator.ModeKeys.TRAIN:
        loss, train_op, input, output = model.getResult(mode)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
        loss = model.getResult(mode)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss)
    else:
        sequence = model.getResult(mode)
        return tf.estimator.EstimatorSpec(mode=mode, predictions=sequence)


dataHelper = DataHelper(os.path.join(FLAGS.dataPath, 'vocabs'))
with open('model/params') as reader:
    params = json.load(reader)
    params = {'hiddenSize': params["hidden_size"],
              'embeddingSize': params["embedding_size"],
              'dropout': params["dropout"],
              'l2Regularizer': params["l2_regularizer"],
              'baseLearnRate': params["base_learnRate"],
              'maxLength': params["max_length"],
              'startToken': dataHelper.vocab2index['<s>'],
              'endToken': dataHelper.vocab2index['</s>'],
              'vocabs': len(dataHelper.vocab2index),
              'beamSearch': FLAGS.beam_search,
              'beamSize': FLAGS.beam_size,
              'encodeLayerSize': FLAGS.encode_layer_size
              }
model = tf.estimator.Estimator(model_fn=model_fn, model_dir='model', params=params)


@app.route('/send', methods=['GET', 'POST'])  # 路由
def test_post():
    uplink = request.form.get('uplink')
    uplink = ' '.join(list(uplink)) + '\n'
    with open('./temp.txt', mode='w') as writer:
        writer.write(uplink)
    test_inputFun = functools.partial(dataHelper.input_fn, os.path.join('.', 'temp.txt'),
                                      os.path.join('.', 'temp.txt'), batch_size=1, is_shuffle_and_repeat=False)
    predictions = model.predict(test_inputFun)
    down_links = list()
    for result in predictions:
        print(dataHelper.dataTransform(result['up_link']))
        if FLAGS.beam_search:
            for line in range(FLAGS.beam_size):
                print(dataHelper.dataTransform(result['down_link'][line]))
                down_links.append(dataHelper.dataTransform(result['down_link'][line]))
        else:
            print(dataHelper.dataTransform(result['down_link']))
    links = {'uplink': "上联:  " + ''.join(uplink.split()), 'downlink': down_links}
    return render_template('index.html', **links)


@app.route('/')
def hello_world():
    return render_template('index.html')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    app.run(host='0.0.0.0',  # 任何ip都可以访问
            port=7777,  # 端口
            debug=True)
