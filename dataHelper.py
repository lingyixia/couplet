# coding:utf-8

# -------------------------------------------------------------------------------
# @Author        chenfeiyu01
# @Name:         dataHelper.py
# @Project       couplet
# @Product       PyCharm
# @DateTime:     2019-06-26 20:54
# @Contact       chenfeiyu01@baidu.com
# @Version       1.0
# @Description:  
# -------------------------------------------------------------------------------
import functools
import tensorflow as tf
from pathlib import Path
tf.enable_eager_execution()

class DataHelper(object):
    def __init__(self, vocab_file):
        self.__getVocab(vocab_file)

    def __getVocab(self, voab_file):
        self.vocab2index = dict()
        self.index2vocab = dict()
        self.vocab2index['<unk>'] = 0
        self.index2vocab[0] = '<unk>'
        self.vocab2index['<pad>'] = 1
        self.index2vocab[1] = '<pad>'
        offset = len(self.vocab2index)
        with Path(voab_file).open() as reader:
            for index, line in enumerate(reader):
                line = line.strip('\n')
                self.index2vocab[index + offset] = line
                self.vocab2index[line] = index + offset

    def parse_fn(self, up_link_line, down_link_line):
        up_link = up_link_line.strip('\n').split()
        # up_link = ['<s>'] + up_link
        # up_link += ['</s>']
        down_link = down_link_line.strip('\n').split()
        down_link = ['<s>'] + down_link
        down_link = down_link + ['</s>']
        up_link = list(map(lambda x: self.vocab2index[x], up_link))
        down_link = list(map(lambda x: self.vocab2index[x], down_link))
        return (up_link, len(up_link)), (down_link, len(down_link)-1)#下联长度减一是因为训练解码阶段输入去掉</s>,输出去掉<s>

    def generator_fn(self, up_link_file, down_link_file):
        with Path(up_link_file).open('r') as up_line_reader, Path(down_link_file).open('r') as down_line_reader:
            for up_link_line, down_link_line in zip(up_line_reader, down_line_reader):
                yield self.parse_fn(up_link_line, down_link_line)

    def input_fn(self, up_link_file, down_link_file, epoch_num=1, batch_size=16, is_shuffle_and_repeat=True):
        shapes = (([None], ()), ([None], ()))
        types = ((tf.int32, tf.int32), (tf.int32, tf.int32))
        defaults = ((self.vocab2index['<pad>'], 0), (self.vocab2index['<pad>'], 0))
        dataset = tf.data.Dataset.from_generator(
            functools.partial(self.generator_fn, up_link_file, down_link_file),
            output_shapes=shapes,
            output_types=types)
        if is_shuffle_and_repeat:
            dataset = dataset.shuffle(1000).repeat(epoch_num)
        dataset = (dataset.padded_batch(batch_size, shapes, defaults).prefetch(1))
        return dataset

    def dataTransform(self, line):
        result = ''
        for i in line:
            if self.index2vocab[i] == '</s>':
                break
            elif self.index2vocab[i] == '<s>':
                continue
            else:
                result += self.index2vocab[i]
        return result


if __name__ == '__main__':
    dataHelper = DataHelper('couplet/vocabs')
    dataset = dataHelper.input_fn('couplet/train/in.txt', 'couplet/train/out.txt',is_shuffle_and_repeat=False)
    for d in dataset:
        print(d)
