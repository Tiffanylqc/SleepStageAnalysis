"""
adapted from https://github.com/MousaviSajad/SleepEEGNet
"""
import dataloader
from seq2seq_sleep_sleep_EDF import build_whole_model, evaluate_metrics
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, f1_score
import random
import time
import os
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split
from tensorflow.python.layers.core import Dense
from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder
from dataloader import SeqDataLoader
import argparse
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False


def load_npz_file(npz_file):
    """Load data_2013 and labels from a npz file."""
    with np.load(npz_file) as f:
        data = f["x"]
        labels = f["y"]
        sampling_rate = f["fs"]
    return data, labels, sampling_rate


def load_score_file(npz_file, seq_len=10):
    """Load the file to be scored"""
    data = []
    labels = []
    tmp_data, tmp_labels, sampling_rate = load_npz_file(npz_file)
    tmp_data = np.squeeze(tmp_data)
    tmp_data = tmp_data.astype(np.float32)
    tmp_labels = tmp_labels.astype(np.int32)
    tmp_data = (
        tmp_data - np.expand_dims(tmp_data.mean(axis=1), axis=1)
    ) / np.expand_dims(tmp_data.std(axis=1), axis=1)
    data.append(tmp_data)
    labels.append(tmp_labels)

    data_score = np.vstack(data)
    label_score = np.hstack(labels)
    data_score = [
        data_score[i : i + seq_len] for i in range(0, len(data_score), seq_len)
    ]
    label_score = [
        label_score[i : i + seq_len] for i in range(0, len(label_score), seq_len)
    ]
    if data_score[-1].shape[0] != seq_len:
        data_score.pop()
        label_score.pop()

    data_score = np.asarray(data_score)
    label_score = np.asarray(label_score)

    return data_score, label_score


def batch_data(x, y, batch_size):
    shuffle = np.random.permutation(len(x))
    start = 0
    x = x[shuffle]
    y = y[shuffle]
    while start + batch_size <= len(x):
        yield x[start : start + batch_size], y[start : start + batch_size]
        start += batch_size


def predict(hparams, FLAGS, filename):
    def evaluate_model(hparams, X_test, y_test, classes):
        acc_track = []
        n_classes = len(classes)
        y_true = []
        y_pred = []
        alignments_alphas_all = []  # (batch_num,B,max_time_step,max_time_step)
        for batch_i, (source_batch, target_batch) in enumerate(
            batch_data(X_test, y_test, hparams.batch_size)
        ):
            # if source_batch.shape[1] != hparams.max_time_step:
            #     print ("Num of steps is: ", source_batch.shape[1])
            # try:
            pred_outputs_ = sess.run(
                pred_outputs, feed_dict={inputs: source_batch, keep_prob_: 1.0}
            )

            alignments_alphas = sess.run(
                dec_states.alignment_history.stack(),
                feed_dict={
                    inputs: source_batch,
                    dec_inputs: target_batch[:, :-1],
                    keep_prob_: 1.0,
                },
            )

            # acc_track.append(np.mean(dec_input == target_batch))
            pred_outputs_ = pred_outputs_[
                :, : hparams.max_time_step
            ]  # remove the last prediction <EOD>
            target_batch_ = target_batch[
                :, 1:-1
            ]  # remove the last <EOD> and the first <SOD>
            acc_track.append(pred_outputs_ == target_batch_)

            alignments_alphas = alignments_alphas.transpose((1, 0, 2))
            alignments_alphas = alignments_alphas[:, : hparams.max_time_step]
            alignments_alphas_all.append(alignments_alphas)

            _y_true = target_batch_.flatten()
            _y_pred = pred_outputs_.flatten()

            y_true.extend(_y_true)
            y_pred.extend(_y_pred)

        cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
        ck_score = cohen_kappa_score(y_true, y_pred)
        acc_avg, acc, f1_macro, f1, sensitivity, specificity, PPV = evaluate_metrics(
            cm, classes
        )
        print (
            "Average Accuracy -> {:>6.4f}, Macro F1 -> {:>6.4f} and Cohen's Kappa -> {:>6.4f} on test set".format(
                acc_avg, f1_macro, ck_score
            )
        )
        for index_ in range(n_classes):
            print (
                "\t{} rhythm -> Sensitivity: {:1.4f}, Specificity: {:1.4f}, Precision (PPV): {:1.4f}, F1 : {:1.4f} Accuracy: {:1.4f}".format(
                    classes[index_],
                    sensitivity[index_],
                    specificity[index_],
                    PPV[index_],
                    f1[index_],
                    acc[index_],
                )
            )
        print (
            "\tAverage -> Sensitivity: {:1.4f}, Specificity: {:1.4f}, Precision (PPV): {:1.4f}, F1-score: {:1.4f}, Accuracy: {:1.4f}".format(
                np.mean(sensitivity),
                np.mean(specificity),
                np.mean(PPV),
                np.mean(f1),
                np.mean(acc),
            )
        )

        return acc_avg, f1_macro, ck_score, y_true, y_pred, alignments_alphas_all

    # num_folds = FLAGS.num_folds
    data_dir = FLAGS.data_dir
    # output_dir = FLAGS.output_dir
    classes = FLAGS.classes
    n_classes = len(classes)

    npz_file = os.path.join(FLAGS.data_dir, filename)
    X_test, y_test = load_score_file(npz_file, seq_len=hparams.max_time_step)
    # preprocessing
    char2numY = dict(zip(classes, range(len(classes))))
    pre_f1_macro = 0

    # <SOD> is a token to show start of decoding  and <EOD> is a token to indicate end of decoding
    char2numY["<SOD>"] = len(char2numY)
    char2numY["<EOD>"] = len(char2numY)
    num2charY = dict(zip(char2numY.values(), char2numY.keys()))

    y_test = [
        [char2numY["<SOD>"]] + [y_ for y_ in date] + [char2numY["<EOD>"]]
        for date in y_test
    ]
    y_test = np.array(y_test)

    with tf.Graph().as_default(), tf.Session() as sess:
        # Placeholders
        inputs = tf.placeholder(
            tf.float32,
            [None, hparams.max_time_step, hparams.input_depth],
            name="inputs",
        )
        targets = tf.placeholder(tf.int32, (None, None), "targets")
        dec_inputs = tf.placeholder(tf.int32, (None, None), "decoder_inputs")
        keep_prob_ = tf.placeholder(tf.float32, name="keep")

        # model
        logits, pred_outputs, loss, optimizer, dec_states = build_whole_model(
            hparams, char2numY, inputs, targets, dec_inputs, keep_prob_
        )
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        saver = tf.train.Saver()
        print (str(datetime.now()))
        # restore the trained model
        ckpt_name = "seq2seqmodel.ckpt"
        ckpt_name = os.path.join(FLAGS.checkpoint_dir, ckpt_name)
        saver.restore(sess, ckpt_name)
        print "Model restored from: {}\n".format(ckpt_name)
        (
            acc_avg,
            f1_macro,
            ck_score,
            y_true,
            y_pred,
            alignments_alphas_all,
        ) = evaluate_model(hparams, X_test, y_test, classes)
        print "Scoring signal input length is: {}\n".format(X_test.shape)
        print "True signal output length is {}\n".format(len(y_true))
        print "True signal output is ", y_true
        print "Predicted signal output length is {}\n".format(len(y_pred))
        print "Predicted signal output is", y_pred
    return y_pred


def main(args=None):

    FLAGS = tf.app.flags.FLAGS

    # outputs_eeg_fpz_cz
    tf.app.flags.DEFINE_string(
        "data_dir", "eeg_data/eeg_fpz_cz", """Directory where to load signal""",
    )
    tf.app.flags.DEFINE_list("classes", ["W", "N1", "N2", "N3", "REM"], """classes""")
    tf.app.flags.DEFINE_string(
        "checkpoint_dir", "./seq2seq-model/", """Directory to load model""",
    )

    # hyperparameters
    hparams = tf.contrib.training.HParams(
        epochs=120,  # 300
        batch_size=20,  # 10
        num_units=128,
        embed_size=10,
        input_depth=3000,
        n_channels=100,
        bidirectional=False,
        use_attention=True,
        lstm_layers=2,
        attention_size=64,
        beam_width=4,
        use_beamsearch_decode=False,
        max_time_step=10,  # 5 3 second best 10# 40 # 100
        output_max_length=10 + 2,  # max_time_step +1
        akara2017=True,
        test_step=5,  # each 10 epochs
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename", type=str, default="ST7022J0.npz", help="File to analyze.",
    )
    args = parser.parse_args()
    label_pred = predict(hparams, FLAGS, args.filename)


if __name__ == "__main__":
    tf.app.run()
