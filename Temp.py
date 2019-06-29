# coding:utf-8

# -------------------------------------------------------------------------------
# @Author        chenfeiyu01
# @Name:         Temp.py
# @Project       couplet
# @Product       PyCharm
# @DateTime:     2019-06-27 11:27
# @Contact       chenfeiyu01@baidu.com
# @Version       1.0
# @Description:  
# -------------------------------------------------------------------------------
import tensorflow as tf
from pathlib import Path
import functools

sess = tf.InteractiveSession()


# tf.enable_eager_execution()


# dataset = tf.data.Dataset.range(10)
# dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
# dataset = dataset.repeat(2)
# dataset = dataset.padded_batch(4, padded_shapes=[None], padding_values=tf.constant(1, dtype=tf.int64))
# 
# iterator = dataset.make_one_shot_iterator()
# next_element = iterator.get_next()
# 
# print(sess.run(next_element))  # ==> [[0, 0, 0], [1, 0, 0], [2, 2, 0], [3, 3, 3]]
# print(sess.run(next_element))  # ==> [[4, 4, 4, 4, 0, 0, 0],
# print(sess.run(next_element))  # ==> [[4, 4, 4, 4, 0, 0, 0],
# print(sess.run(next_element))  # ==> [[4, 4, 4, 4, 0, 0, 0],
# print(sess.run(next_element))  # ==> [[4, 4, 4, 4, 0, 0, 0],


def parse_fn(up_link_line, down_link_line):
    up_link = up_link_line.strip('\n').split()
    up_link = ['<s>'] + up_link
    up_link += ['</s>']
    down_link = down_link_line.strip('\n').split()
    down_link_input = ['<s>'] + down_link
    down_link_ouput = down_link + ['</s>']
    up_link = list(map(lambda x: vocab2index[x], up_link))
    down_link_input = list(map(lambda x: vocab2index[x], down_link_input))
    down_link_ouput = list(map(lambda x: vocab2index[x], down_link_ouput))
    return (up_link, len(up_link)), (down_link_input, down_link_ouput, len(down_link_ouput))


def generator_fn(up_link_file, down_link_file):
    with Path(up_link_file).open('r') as up_line_reader, Path(down_link_file).open('r') as down_line_reader:
        for up_link_line, down_link_line in zip(up_line_reader, down_line_reader):
            yield parse_fn(up_link_line, down_link_line)


def input_fn(up_link_file, down_link_file, epoch_num=1, batch_size=5, is_shuffle_and_repeat=True):
    shapes = (([None], ()), ([None], [None], ()))
    types = ((tf.int32, tf.int32), (tf.int32, tf.int32, tf.int32))
    defaults = ((vocab2index['<pad>'], 0), (vocab2index['<pad>'], vocab2index['<pad>'], 0))
    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator_fn, up_link_file, down_link_file),
        output_shapes=shapes,
        output_types=types)
    if is_shuffle_and_repeat:
        dataset = dataset.shuffle(1000).repeat(epoch_num)
    dataset = (dataset.padded_batch(batch_size, shapes, defaults).prefetch(1))
    dataset = dataset.make_one_shot_iterator()
    return dataset.get_next()


if __name__ == '__main__':
    # vocabs = set()
    # result = list()
    # vocabs.add('<s>')
    # vocabs.add('</s>')
    # vocabs.add('。')
    # with Path('couplet/test/in.txt').open() as reader:
    #     for line in reader:
    #         line = line.strip()
    #         vocabs |= set(line.split(' '))
    # with Path('couplet/test/out.txt').open() as reader:
    #     for line in reader:
    #         line = line.strip()
    #         vocabs |= set(line.split(' '))
    # vocabs = list(vocabs)
    # for line in vocabs:
    #     result.append(line + '\n')
    # with Path('vocabs').open(mode='w') as writer:
    #     writer.writelines(result)
    # a = tf.Variable(initial_value=[[1, 2, 3], [4, 5, 6]])
    # b = tf.where(tf.not_equal(a,2))
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     print(sess.run(b))
    print('s', end='')
    print()
    print('s')
