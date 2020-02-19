import tensorflow as tf
import os


def config_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            # os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # 指定第一块GPU可用

        except RuntimeError as e:
            print(e)


def config_gpudd():
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if not gpus:
        print("______________________")

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

        tf.config.experimental.set_virtual_device_configuration(
            gpu,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)]
        )
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

        os.environ["CUDA_VISIBLE_DEVICES"] = "1"






if __name__ == '__main__':
    config_gpu()