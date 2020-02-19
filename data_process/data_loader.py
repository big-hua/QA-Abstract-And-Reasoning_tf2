import pandas as pd
from utils.multi_proc_utils import parallelize
from data_process.file_utils import save_dict
from data_process.data_func import *
from data_process.wv_loader import Vocab
from utils.config import train_seg_path, test_seg_path, merger_seg_path, user_dict, train_x_seg_path, test_x_seg_path, \
    train_x_pad_path, train_y_pad_path, test_x_pad_path, wv_train_epochs, embedding_matrix_path, \
    vocab_path, reverse_vocab_path, train_x_path, train_y_path, test_x_path, embedding_dim, train_y_seg_path, \
    val_x_seg_path, val_y_seg_path,stop_word_path, train_data_path, test_data_path,save_wv_model_path

from gensim.models.word2vec import LineSentence, Word2Vec
from sklearn.model_selection import train_test_split
# 引入日志配置
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# 自定义词表
jieba.load_userdict(user_dict)
# 加载停用词
stop_words = load_stop_words(stop_word_path)

def build_dataset(train_data_path, test_data_path):
    '''
    数据加载+预处理
    :param train_data_path:训练集路径
    :param test_data_path: 测试集路径
    :return: 训练数据 测试数据  合并后的数据
    '''
    # 1.加载数据
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)
    print('train data size {},test data size {}'.format(len(train_df), len(test_df)))

    # 2. 空值剔除
    train_df.dropna(subset=['Report'], inplace=True)

    train_df.fillna('', inplace=True)
    test_df.fillna('', inplace=True)

    # 3.多线程, 批量数据处理
    train_df = parallelize(train_df, sentences_proc)
    test_df = parallelize(test_df, sentences_proc)

    # 4. 合并训练测试集合
    train_df['merged'] = train_df[['Question', 'Dialogue', 'Report']].apply(lambda x: ' '.join(x), axis=1)
    test_df['merged'] = test_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    merged_df = pd.concat([train_df[['merged']], test_df[['merged']]], axis=0)
    print('train data size {},test data size {},merged_df data size {}'.format(len(train_df),
                                                                               len(test_df),
                                                                               len(merged_df)))

    # 5.保存处理好的 训练 测试集合
    train_df = train_df.drop(['merged'], axis=1)
    test_df = test_df.drop(['merged'], axis=1)

    train_df.to_csv(train_seg_path, index=None, header=False)
    test_df.to_csv(test_seg_path, index=None, header=False)

    # 6. 保存合并数据
    merged_df.to_csv(merger_seg_path, index=None, header=False)

    # 7. 训练词向量
    print('start build w2v model')
    wv_model = Word2Vec(LineSentence(merger_seg_path),
                        size=embedding_dim,
                        sg=1,
                        workers=8,
                        iter=wv_train_epochs,
                        window=5,
                        min_count=5)

    # 8. 分离数据和标签
    train_df['X'] = train_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    test_df['X'] = test_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)

    # 训练集 验证集划分
    X_train, X_val, y_train, y_val = train_test_split(train_df['X'], train_df['Report'],
                                                      test_size=0.002,  # 8W*0.002
                                                      )

    X_train.to_csv(train_x_seg_path, index=None, header=False)
    y_train.to_csv(train_y_seg_path, index=None, header=False)
    X_val.to_csv(val_x_seg_path, index=None, header=False)
    y_val.to_csv(val_y_seg_path, index=None, header=False)

    test_df['X'].to_csv(test_x_seg_path, index=None, header=False)

    # 9. 填充开始结束符号,未知词填充 oov, 长度填充
    # 使用GenSim训练得出的vocab
    vocab = wv_model.wv.vocab

    # 训练集X处理
    # 获取适当的最大长度
    train_x_max_len = get_max_len(train_df['X'])
    test_X_max_len = get_max_len(test_df['X'])
    X_max_len = max(train_x_max_len, test_X_max_len)
    train_df['X'] = train_df['X'].apply(lambda x: pad_proc(x, X_max_len, vocab))

    # 测试集X处理
    # 获取适当的最大长度
    test_df['X'] = test_df['X'].apply(lambda x: pad_proc(x, X_max_len, vocab))

    # 训练集Y处理
    # 获取适当的最大长度
    train_y_max_len = get_max_len(train_df['Report'])
    train_df['Y'] = train_df['Report'].apply(lambda x: pad_proc(x, train_y_max_len, vocab))

    # 10. 保存pad oov处理后的,数据和标签
    train_df['X'].to_csv(train_x_pad_path, index=None, header=False)
    train_df['Y'].to_csv(train_y_pad_path, index=None, header=False)
    test_df['X'].to_csv(test_x_pad_path, index=None, header=False)

    print('train_x_max_len:{} ,train_y_max_len:{}'.format(X_max_len, train_y_max_len))

    # 11. 词向量再次训练
    print('start retrain w2v model')
    wv_model.build_vocab(LineSentence(train_x_pad_path), update=True)
    wv_model.train(LineSentence(train_x_pad_path), epochs=1, total_examples=wv_model.corpus_count)

    print('1/3')
    wv_model.build_vocab(LineSentence(train_y_pad_path), update=True)
    wv_model.train(LineSentence(train_y_pad_path), epochs=1, total_examples=wv_model.corpus_count)

    print('2/3')
    wv_model.build_vocab(LineSentence(test_x_pad_path), update=True)
    wv_model.train(LineSentence(test_x_pad_path), epochs=1, total_examples=wv_model.corpus_count)

    # 保存词向量模型
    wv_model.save(save_wv_model_path)
    print('finish retrain w2v model')
    print('final w2v_model has vocabulary of ', len(wv_model.wv.vocab))

    # 12. 更新vocab
    vocab = {word: index for index, word in enumerate(wv_model.wv.index2word)}
    reverse_vocab = {index: word for index, word in enumerate(wv_model.wv.index2word)}

    # 保存字典
    save_dict(vocab_path, vocab)
    save_dict(reverse_vocab_path, reverse_vocab)

    # 13. 保存词向量矩阵
    embedding_matrix = wv_model.wv.vectors
    np.save(embedding_matrix_path, embedding_matrix)

    # 14. 数据集转换 将词转换成索引  [<START> 方向机 重 ...] -> [32800, 403, 986, 246, 231
    vocab = Vocab(vocab_path)

    train_ids_x = train_df['X'].apply(lambda x: transform_data(x, vocab.word2id))
    train_ids_y = train_df['Y'].apply(lambda x: transform_data(x, vocab.word2id))
    test_ids_x = test_df['X'].apply(lambda x: transform_data(x, vocab.word2id))

    # 15. 数据转换成numpy数组
    # 将索引列表转换成矩阵 [32800, 403, 986, 246, 231] --> array([[32800,   403,   986 ]]
    train_X = np.array(train_ids_x.tolist())
    train_Y = np.array(train_ids_y.tolist())
    test_X = np.array(test_ids_x.tolist())

    # 保存数据
    np.save(train_x_path, train_X)
    np.save(train_y_path, train_Y)
    np.save(test_x_path, test_X)
    return train_X, train_Y, test_X

def load_dataset(max_enc_len=400, max_dec_len=60):
    """
    :return: 加载处理好的数据集
    """
    train_X = np.load(train_x_path + '.npy')
    train_Y = np.load(train_y_path + '.npy')
    test_X = np.load(test_x_path + '.npy')

    train_X = train_X[:, :max_enc_len]
    train_Y = train_Y[:, :max_dec_len]
    test_X = test_X[:, :max_enc_len]
    return train_X, train_Y, test_X


def load_train_dataset(max_enc_len, max_dec_len):
    """
    :return: 加载处理好的数据集
    """
    train_X = np.load(train_x_path + '.npy')
    train_Y = np.load(train_y_path + '.npy')

    train_X = train_X[:, :max_enc_len]
    train_Y = train_Y[:, :max_dec_len]
    return train_X, train_Y


def load_test_dataset(max_enc_len=400):
    """
    :return: 加载处理好的数据集
    """
    test_X = np.load(test_x_path + '.npy')
    test_X = test_X[:, :max_enc_len]
    return test_X


if __name__ == '__main__':
    # 数据集批量处理
    build_dataset(train_data_path, test_data_path)


