import tensorflow as tf
from seq2seq.seq2seq_model import Seq2Seq
from seq2seq.train_helper import train_model
from utils.config import save_wv_model_path, checkpoint_dir
from utils.params_utils import get_params
from data_process.wv_loader import get_vocab


def train(params):
    # GPU资源配置
    # config_gpu()

    # 读取vocab训练
    vocab, _ = get_vocab(save_wv_model_path)

    params['vocab_size'] = len(vocab)

    # 构建模型
    print("Building the model ...")
    model = Seq2Seq(params)

    # 获取保存管理者
    checkpoint = tf.train.Checkpoint(Seq2Seq=model)

    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

    # 训练模型
    train_model(model, vocab, params, checkpoint_manager)


if __name__ == '__main__':
    # 获得参数
    params = get_params()
    # 训练模型
    train(params)
