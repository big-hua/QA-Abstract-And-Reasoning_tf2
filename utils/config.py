import os
import pathlib

# 获取项目根目录
root = pathlib.Path(os.path.abspath(__file__)).parent.parent

# 原有数据路径
train_data_path = os.path.join(root, 'data', 'AutoMaster_TrainSet.csv')
test_data_path = os.path.join(root, 'data', 'AutoMaster_TestSet.csv')
stop_word_path = os.path.join(root, 'data', 'stopwords/stopwords.txt')
user_dict = os.path.join(root, 'data', 'user_dict.txt')

# 预处理后的数据
train_seg_path = os.path.join(root, 'data', 'train_seg_data.csv')
test_seg_path = os.path.join(root, 'data', 'test_seg_data.csv')
merger_seg_path = os.path.join(root, 'data', 'merged_train_test_seg_data.csv')

# 数据标签分离
train_x_seg_path = os.path.join(root, 'data', 'train_X_seg_data.csv')
train_y_seg_path = os.path.join(root, 'data', 'train_Y_seg_data.csv')
val_x_seg_path = os.path.join(root, 'data', 'val_X_seg_data.csv')
val_y_seg_path = os.path.join(root, 'data', 'val_Y_seg_data.csv')
test_x_seg_path = os.path.join(root, 'data', 'test_X_seg_data.csv')

# 数据标签分离，pad oov处理后的数据
train_x_pad_path = os.path.join(root, 'data', 'train_x_pad_path.csv')
train_y_pad_path = os.path.join(root, 'data', 'train_y_pad_path.csv')
test_x_pad_path = os.path.join(root, 'data', 'test_x_pad_path.csv')

# numpy 转换后的数据
train_x_path = os.path.join(root, 'data', 'train_X')
train_y_path = os.path.join(root, 'data', 'train_Y')
test_x_path = os.path.join(root, 'data', 'test_X')


# 词向量模型路径
save_wv_model_path = os.path.join(root, 'data', 'wv/word2vec.model')
# 词向量矩阵路径
embedding_matrix_path = os.path.join(root, 'data', 'wv/embedding_matrix')
# 字典路径
vocab_path = os.path.join(root, 'data', 'wv/vocab.txt')
reverse_vocab_path = os.path.join(root, 'data', 'wv/reverse_vocab.txt')


# 词向量模型
checkpoint_dir = os.path.join(root, 'data', 'checkpoints', 'training_checkpoints_pgn_cov_backed')
seq2seq_checkpoint_dir = os.path.join(root, 'data', 'checkpoints', 'training_checkpoints_seq2seq')
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

# 结果保存文件夹
save_result_dir = os.path.join(root, 'result')

# 迭代次数
wv_train_epochs = 10

# 词向量维度
embedding_dim = 300

sample_total = 82871

batch_size = 32

epochs = 10

vocab_size = 50000



