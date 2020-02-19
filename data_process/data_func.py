import re
import jieba
from utils.config import stop_word_path

import numpy as np


def get_max_len(data):
    """
    获得合适的最大长度值
    :param data: 待统计的数据  train_df['Question']
    :return: 最大长度值
    """
    max_lens = data.apply(lambda x: x.count(' ') + 1)
    return int(np.mean(max_lens) + 2 * np.std(max_lens))


def pad_proc(sentence, max_len, vocab):
    '''
    < start > < end > < pad > < unk > max_lens
    '''
    # 0.按空格统计切分出词
    words = sentence.strip().split(' ')
    # 1. 截取规定长度的词数
    words = words[:max_len]
    # 2. 填充< unk > ,判断是否在vocab中, 不在填充 < unk >
    sentence = [word if word in vocab else '<UNK>' for word in words]
    # 3. 填充< start > < end >
    sentence = ['<START>'] + sentence + ['<STOP>']
    # 4. 判断长度，填充　< pad >
    sentence = sentence + ['<PAD>'] * (max_len - len(words))
    return ' '.join(sentence)


def load_stop_words(stop_word_path):
    '''
    加载停用词
    :param stop_word_path:停用词路径
    :return: 停用词表 list
    '''
    # 打开文件
    file = open(stop_word_path, 'r', encoding='utf-8')
    # 读取所有行
    stop_words = file.readlines()
    # 去除每一个停用词前后 空格 换行符
    stop_words = [stop_word.strip() for stop_word in stop_words]
    return stop_words





def clean_sentence(sentence):
    '''
    特殊符号去除
    :param sentence: 待处理的字符串
    :return: 过滤特殊字符后的字符串
    '''
    if isinstance(sentence, str):
        return re.sub(
            # r'[\s+\-\!\/\|\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】“”！，。？、~@#￥%……&*（）]+|车主说|技师说|语音|图片|你好|您好',
            r'[\s+\-\!\/\|\[\]\{\}_,$%^*(+\"\')]+|[:：+——()?【】“”~@#￥%……&*（）]+|车主说|技师说|语音|图片',
            '', sentence)
    else:
        return ' '


def filter_stopwords(words, stop_words):
    '''
    过滤停用词
    :param seg_list: 切好词的列表 [word1 ,word2 .......]
    :return: 过滤后的停用词
    '''
    return [word for word in words if word not in stop_words]


def sentence_proc(sentence):
    '''
    预处理模块
    :param sentence:待处理字符串
    :return: 处理后的字符串
    '''
    # 清除无用词
    # sentence = clean_sentence(sentence)
    # 切词，默认精确模式，全模式cut参数cut_all=True
    words = jieba.cut(sentence)
    # 过滤停用词
    # words = filter_stopwords(words)
    # 拼接成一个字符串,按空格分隔
    return ' '.join(words)


def sentences_proc(df):
    '''
    数据集批量处理方法
    :param df: 数据集
    :return:处理好的数据集
    '''
    # 批量预处理 训练集和测试集
    for col_name in ['Brand', 'Model', 'Question', 'Dialogue']:
        df[col_name] = df[col_name].apply(sentence_proc)

    if 'Report' in df.columns:
        # 训练集 Report 预处理
        df['Report'] = df['Report'].apply(sentence_proc)
    return df


def transform_data(sentence, vocab):
    """
    word 2 index
    :param sentence: [word1,word2,word3, ...] ---> [index1,index2,index3 ......]
    :param vocab: 词表
    :return: 转换后的序列
    """
    # 字符串切分成词
    words = sentence.split(' ')
    # 按照vocab的index进行转换         # 遇到未知词就填充unk的索引
    ids = [vocab[word] if word in vocab else vocab['<UNK>'] for word in words]
    return ids


def preprocess_sentence(sentence, max_len, vocab):
    """
    单句话预处理
    """
    # 1. 切词处理
    sentence = sentence_proc(sentence)
    # 2. 填充
    sentence = pad_proc(sentence, max_len - 2, vocab)
    # 3. 转换index
    sentence = transform_data(sentence, vocab)
    return np.array([sentence])