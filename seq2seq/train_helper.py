import tensorflow as tf

from seq2seq.batcher import train_batch_generator
from pgn.batcher import batcher
from seq2seq.loss import loss_function
import time


def train_model(model, vocab, params, checkpoint_manager):
    epochs = params['epochs']
    batch_size = params['batch_size']

    pad_index = vocab.word2id[vocab.PAD_TOKEN]
    start_index = vocab.word2id[vocab.START_DECODING]

    # 计算vocab size
    params['vocab_size'] = vocab.count

    optimizer = tf.keras.optimizers.Adam(name='Adam', learning_rate=0.01)

    # 训练
    @tf.function(input_signature=(tf.TensorSpec(shape=[params["batch_size"], None], dtype=tf.int32),
                                  tf.TensorSpec(shape=[params["batch_size"], params["max_dec_len"]], dtype=tf.int32),
                                  tf.TensorSpec(shape=[params["batch_size"], params["max_dec_len"]], dtype=tf.int32)))
    def train_step(enc_inp, dec_target, dec_input):
        batch_loss = 0
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = model.call_encoder(enc_inp)
            # 第一个隐藏层输入
            dec_hidden = enc_hidden
            # 逐个预测序列
            predictions, _ = model(dec_input, dec_hidden, enc_output, dec_target)

            batch_loss = loss_function(dec_target, predictions, pad_index)

            variables = model.encoder.trainable_variables + model.decoder.trainable_variables + model.attention.trainable_variables

            gradients = tape.gradient(batch_loss, variables)

            optimizer.apply_gradients(zip(gradients, variables))

            return batch_loss

    dataset = batcher(vocab, params)

    for epoch in range(epochs):
        start = time.time()
        total_loss = 0

        batch = 0
        for encoder_batch_data, decoder_batch_data in dataset:
            inputs = encoder_batch_data['enc_input']
            target = decoder_batch_data['dec_target']
            dec_input = decoder_batch_data['dec_input']

            # for (batch, (inputs, target)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inputs, target, dec_input)
            total_loss += batch_loss

            if batch % 50 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
            batch += 1
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            ckpt_save_path = checkpoint_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / batch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
