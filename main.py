import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from config import cfg
#from utils import load_data
from capsNet import CapsNet
import data_helpers
from tensorflow.contrib import learn


def preprocess():
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    x_text, y = data_helpers.load_data_and_labels(cfg.positive_data_file, cfg.negative_data_file)

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = int(0.8 * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    print("Train/Dev split: {:d},{:d}".format(len(x_train[0]), len(x_dev[0])))
    return x_train, y_train, vocab_processor, x_dev, y_dev


def save_to():
    if not os.path.exists(cfg.results):
        os.mkdir(cfg.results)
    if cfg.is_training:
        loss = cfg.results + '/loss.csv'
        train_acc = cfg.results + '/train_acc.csv'
        val_acc = cfg.results + '/val_acc.csv'
        val_loss = cfg.results + '/val_loss.csv'
        if os.path.exists(val_acc):
            os.remove(val_acc)
        if os.path.exists(loss):
            os.remove(loss)
        if os.path.exists(train_acc):
            os.remove(train_acc)
        if os.path.exists(val_loss):
            os.remove(val_loss)

        fd_train_acc = open(train_acc, 'w')
        fd_train_acc.write('step,train_acc\n')
        fd_loss = open(loss, 'w')
        fd_loss.write('step,loss\n')
        fd_val_acc = open(val_acc, 'w')
        fd_val_acc.write('step,val_acc\n')
        fd_val_loss = open(val_loss, 'w')
        fd_val_loss.write('step,val_loss\n')

        return fd_train_acc, fd_loss, fd_val_acc, fd_val_loss
    else:
        test_acc = cfg.results + '/test_acc.csv'
        if os.path.exists(test_acc):
            os.remove(test_acc)
        fd_test_acc = open(test_acc, 'w')
        fd_test_acc.write('test_acc\n')
        return (fd_test_acc)


def train():
    x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()
    # Training loop. For each batch...
    with tf.Graph().as_default():
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            model = CapsNet(sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=cfg.embedding_dim)
            fd_train_acc, fd_loss, fd_val_acc, fd_val_loss = save_to()
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(model.total_loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    model.input_x: x_batch,
                    model.input_y: y_batch,
                }
                _, step, total_loss, accuracy = sess.run([train_op, global_step, model.total_loss, model.accuracy], feed_dict=feed_dict)

                if step % cfg.train_sum_freq == 0:
                    print("start train summary!!!---step={}\n".format(step))
                    print("total_loss = {},accuracy= {}\n".format(total_loss, accuracy))
                    fd_loss.write(str(step) + ',' + str(total_loss) + "\n")
                    fd_loss.flush()
                    fd_train_acc.write(str(step) + ',' + str(accuracy) + "\n")
                    fd_train_acc.flush()


            def dev_step(x_dev, y_dev):
                """
                Evaluates model on a dev set
                """
                batches = data_helpers.dev_iter(
                    list(zip(x_dev, y_dev)), cfg.batch_size)
                total_acc = 0
                total_loss = 0
                count = 0
                step = 0
                for batch in batches:
                    x_batch, y_batch = zip(*batch)
                    feed_dict = {
                        model.input_x: x_batch,
                        model.input_y: y_batch,
                    }
                    step, acc, batch_loss = sess.run([global_step, model.accuracy, model.total_loss],
                                               feed_dict=feed_dict)
                    total_acc += acc
                    total_loss += batch_loss
                    count += 1
                    print("loss:{},acc:{}\n".format(batch_loss, acc))
                total_loss = total_loss / count
                total_acc = total_acc / count
                fd_val_loss.write(str(step) + ',' + str(total_loss) + "\n")
                fd_val_loss.flush()
                fd_val_acc.write(str(step) + ',' + str(total_acc) + '\n')
                fd_val_acc.flush()

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), cfg.batch_size, cfg.epoch)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % cfg.val_sum_freq == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev)
                    print("")


    # with tf.Graph().as_default():
    #     model = CapsNet()
    #     fd_train_acc, fd_loss, fd_val_acc, fd_val_loss = save_to()
    #     config = tf.ConfigProto(allow_soft_placement=True)
    #     config.gpu_options.allow_growth = True
    #     with tf.Session(config=config) as sess:
    #         print("\nNote: all of results will be saved to directory: " + cfg.results)
    #         sess.run(tf.global_variables_initializer())
    #         for epoch in range(cfg.epoch):
    #             print("Training for epoch %d/%d:" % (epoch, cfg.epoch))
    #             for step in tqdm(range(num_tr_batch), total=num_tr_batch, ncols=70, leave=False, unit='b'):
    #                 start = step * cfg.batch_size
    #                 end = start + cfg.batch_size
    #                 global_step = epoch * num_tr_batch + step
    #                 if global_step % cfg.train_sum_freq == 0:
    #                     _, loss, train_acc, summary_str = sess.run(
    #                         [model.train_op, model.total_loss, model.accuracy, model.train_summary],
    #                         {model.X: trX[start:end], model.labels: trY[start:end], model.keep_prob: 1.0})
    #                     assert not np.isnan(loss), 'Something wrong! loss is nan...'
    #                     fd_loss.write(str(global_step) + ',' + str(loss) + "\n")
    #                     fd_loss.flush()
    #                     fd_train_acc.write(str(global_step) + ',' + str(train_acc / cfg.batch_size) + "\n")
    #                     fd_train_acc.flush()
    #                 else:
    #                     sess.run([model.train_op, model.total_loss, model.accuracy],
    #                              {model.X: trX[start:end], model.labels: trY[start:end], model.keep_prob: 0.75})
    #
    #                 if cfg.val_sum_freq != 0 and (global_step) % cfg.val_sum_freq == 0:
    #                     val_acc = 0
    #                     loss = 0
    #                     for i in range(num_val_batch):
    #                         start = i * cfg.batch_size
    #                         end = start + cfg.batch_size
    #                         acc, batch_loss = sess.run([model.accuracy, model.total_loss],
    #                                                    {model.X: valX[start:end], model.labels: valY[start:end],
    #                                                     model.keep_prob: 1.0})
    #                         val_acc += acc
    #                         loss += batch_loss
    #                     val_acc = val_acc / (cfg.batch_size * num_val_batch)
    #                     loss = loss / num_val_batch
    #                     fd_val_loss.write(str(global_step) + ',' + str(loss) + "\n")
    #                     fd_val_loss.flush()
    #                     fd_val_acc.write(str(global_step) + ',' + str(val_acc) + '\n')
    #                     fd_val_acc.flush()
    #
    #             # if (epoch + 1) % cfg.save_freq == 0:
    #             #     supervisor.saver.save(sess, cfg.logdir + '/model_epoch_%04d_step_%02d' % (epoch, global_step))
    #
    #         fd_val_acc.close()
    #         fd_train_acc.close()
    #         fd_loss.close()

#
# def evaluation(model, supervisor, num_label):
#     teX, teY, num_te_batch = load_data(cfg.dataset, cfg.batch_size, is_training=False)
#     fd_test_acc = save_to()
#     with supervisor.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
#         supervisor.saver.restore(sess, tf.train.latest_checkpoint(cfg.logdir))
#         tf.logging.info('Model restored!')
#
#         test_acc = 0
#         for i in tqdm(range(num_te_batch), total=num_te_batch, ncols=70, leave=False, unit='b'):
#             start = i * cfg.batch_size
#             end = start + cfg.batch_size
#             acc = sess.run(model.accuracy,
#                            {model.X: teX[start:end], model.labels: teY[start:end], model.keep_prob: 1.0})
#             test_acc += acc
#         test_acc = test_acc / (cfg.batch_size * num_te_batch)
#         fd_test_acc.write(str(test_acc))
#         fd_test_acc.close()
#         print('Test accuracy has been saved to ' + cfg.results + '/test_acc.csv')


def main(_):
    tf.logging.info(' Loading Graph...')

    tf.logging.info(' Graph loaded')

    # sv = tf.train.Supervisor(graph=model.graph, logdir=cfg.logdir, save_model_secs=0)
    # if cfg.is_training:
    tf.logging.info(' Start training...')
    # train(model, sv, num_label)
    train()
    tf.logging.info('Training done')
    # else:
    #     # evaluation(model, sv, num_label)
    #     evaluation(model, num_label)


if __name__ == "__main__":
    tf.app.run()
