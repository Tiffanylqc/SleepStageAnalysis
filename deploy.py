from flask import Flask, request, Response, jsonify, make_response
import glob
import ntpath
import os
import shutil
import pandas as pd
from mne import Epochs, pick_types, find_events
from mne.io import concatenate_raws, read_raw_edf

from seq2seq_sleep_sleep_EDF import build_whole_model
import eeg_frag_info_combine
import numpy as np
import scipy.io as spio
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split
from tensorflow.python.layers.core import Dense
from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False


app = Flask(__name__)

# Label values
W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
UNKNOWN = 5

stage_dict = {"W": W, "N1": N1, "N2": N2, "N3": N3, "REM": REM, "UNKNOWN": UNKNOWN}
class_dict = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM", 5: "UNKNOWN"}

ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": 5,
    "Movement time": 5,
}

EPOCH_SEC_SIZE = 30

DATA_DIR = "./eeg_data/"
SELECT_CH = "EEG Fpz-Cz"

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


def preprocess_seq2seq(x, seq_len=10):
    data = []
    x = np.squeeze(x)
    x = x.astype(np.float32)
    x = (x - np.expand_dims(x.mean(axis=1), axis=1)) / np.expand_dims(
        x.std(axis=1), axis=1
    )
    data.append(x)
    data_score = np.vstack(data)
    data_score = [
        data_score[i : i + seq_len] for i in range(0, len(data_score), seq_len)
    ]
    if data_score[-1].shape[0] != seq_len:
        data_score.pop()
    data_score = np.asarray(data_score)
    return data_score


def batch_data(x, batch_size):
    start = 0
    while start + batch_size <= len(x):
        yield x[start : start + batch_size]
        start += batch_size


def predict(hparams, X_test):
    def get_y_pred(hparams, X_test, classes, sess, pred_outputs):
        n_classes = len(classes)
        y_pred = []
        for batch_i, (source_batch) in enumerate(
            batch_data(X_test, hparams.batch_size)
        ):
            pred_outputs_ = sess.run(
                pred_outputs, feed_dict={inputs: source_batch, keep_prob_: 1.0}
            )
            pred_outputs_ = pred_outputs_[:, : hparams.max_time_step]
            _y_pred = pred_outputs_.flatten()
            y_pred.extend(_y_pred)
        return y_pred

    classes = ["W", "N1", "N2", "N3", "REM"]
    n_classes = len(classes)
    char2numY = dict(zip(classes, range(len(classes))))
    # <SOD> is a token to show start of decoding  and <EOD> is a token to indicate end of decoding
    char2numY["<SOD>"] = len(char2numY)
    char2numY["<EOD>"] = len(char2numY)
    num2charY = dict(zip(char2numY.values(), char2numY.keys()))
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
        # restore the trained model
        ckpt_name = "seq2seqmodel.ckpt"
        ckpt_name = os.path.join("./seq2seq-model/", ckpt_name)
        saver.restore(sess, ckpt_name)
        y_pred = get_y_pred(hparams, X_test, classes, sess, pred_outputs)
    return y_pred


@app.route("/")
def index():
    return "<h1>Success</h1>"


@app.route("/api/analysis", methods=["POST"])
def sleep_analysis():
    # save uploaded file
    try:
        f = request.files["file"]
        # Output dir
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        else:
            shutil.rmtree(DATA_DIR)
            os.makedirs(DATA_DIR)
        psg_fname = os.path.join(DATA_DIR, f.filename)
        f.save(psg_fname)
    except KeyError:
        return make_response(
            jsonify(message="Sample file 'file' missing in POST request"), 400
        )

    # preprocess sleep file
    raw = read_raw_edf(psg_fname, preload=True, stim_channel=None)
    sampling_rate = raw.info["sfreq"]
    raw_ch_df = raw.to_data_frame(scaling_time=100.0)[SELECT_CH]
    raw_ch_df = raw_ch_df.to_frame()
    raw_ch_df.set_index(np.arange(len(raw_ch_df)))
    raw_ch = raw_ch_df.values
    # Verify that we can split into 30-s epochs
    remainder = len(raw_ch) % (EPOCH_SEC_SIZE * sampling_rate)
    raw_ch = raw_ch[: len(raw_ch) - int(remainder)]
    if len(raw_ch) % (EPOCH_SEC_SIZE * sampling_rate) != 0:
        raise Exception("Something wrong")
    n_epochs = len(raw_ch) / (EPOCH_SEC_SIZE * sampling_rate)
    # Get epochs and their corresponding labels
    x = np.asarray(np.split(raw_ch, n_epochs)).astype(np.float32)

    # predict stage with seq2seq model
    data_score = preprocess_seq2seq(x)
    pred_label = predict(hparams, data_score)
    # print(pred_label)
    pred_stage_name = map(lambda i: class_dict[i], pred_label)
    # overall score, overall msg, total sleep time,
    # weighted transition rateweighted transition rate msg, transition info for each epoch
    (
        score,
        overall_msg,
        tst,
        wtr,
        wtr_msg,
        epoch_msg,
    ) = eeg_frag_info_combine.eeg_frag_info(pred_label, EPOCH_SEC_SIZE)
    global global_section_msg
    global global_signal_val
    global global_pred_stage_name

    global_signal_val = x
    global_section_msg = epoch_msg
    global_pred_stage_name = pred_stage_name

    section_info = {}
    i = 0
    section_name = "section_" + str(i)
    y = np.squeeze(global_signal_val[i]).tolist()
    section_info = {
        "section_id": i,
        "stage_name": global_pred_stage_name[i],
        "description": global_section_msg[i],
        "signal": {"x": range(0, 3000), "y": y,},  # x is the time, unit in seconds
    }
    return make_response(
        jsonify(
            message="Sample file upload succeeds in POST request",
            sleep_score=score,
            wtr=wtr,
            wtr_msg=wtr_msg,
            sleep_msg=overall_msg,
            total_sleep_time=tst,
            first_section_info=section_info,
            total_section_num=len(global_pred_stage_name),
        ),
        200,
    )


@app.route("/api/section")
def get_section_info():
    i = int(request.args.get("sec_id"))
    section_info = {}
    section_name = "section_" + str(i)
    y = np.squeeze(global_signal_val[i]).tolist()
    section_info = {
        "section_id": i,
        "stage_name": global_pred_stage_name[i],
        "description": global_section_msg[i],
        "signal": {"x": range(0, 3000), "y": y,},  # x is the time, unit in seconds
    }
    return make_response(jsonify(section=section_info), 200)


if __name__ == "__main__":
    app.run(debug=True)
