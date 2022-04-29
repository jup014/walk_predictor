import datetime
import json
import os
import sqlite3
from time import sleep
import numpy as np

import pandas as pd
from lib.db import fetch_job_in_queue, post_result
from lib.report import log


from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import xgboost
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
import pickle
from joblib import dump

N_WORKER = 5



def work(k = None):
    log("worker started")
    while True:
        cpu, ram = get_5_sec_cpu_and_ram_usage()
        log("cpu: {}%, mem: {}%".format(cpu, ram))

        
        if k:
            log("fetching job for k={}".format(k))
            job = fetch_job(k)
            # row = retry_db(fetch_job, k)
        else:
            log("  fetching a job")
            job = fetch_job()
            # row = retry_db(fetch_job)
        
        if job is None:
            if k:
                log("  no jobs for k={}. Retrying for any k...".format(k))
                k = None
            else:
                log("  no jobs. Exiting...")
                return
        else:
            job_id = str(job['_id'])
            new_k = job['k']
            
            algorithm = job['algorithm']

            log("  job found: id={}, k={}, algorithm={}".format(job_id, new_k, algorithm))

            # log("  updating job({}) status to 'running'".format(job_id))
            # update_job_status_to_running(job_id)
            # log("  job updated")

            if k:
                if k == new_k:
                    log("  data for k={} is already in memory. Moving on...".format(k))
                    loading_needed = False
                else:
                    log("  another data for k={} is already in memory. Reloading a new data for k={}...".format(k, new_k))
                    k = new_k
                    loading_needed = True
            else:
                log("  There is no data in memory. Loading a new data for k={}...".format(new_k))
                k = new_k
                loading_needed = True

            if loading_needed:
                n_dim, n_classes, train_input_np, train_output_np, test_input_np, test_output_np = data_load_for_k(k)

            if algorithm == "mlp":
                n_layer = job["n_layer"]
                neuron_arch = job["neuron_arch"]
            else:
                n_layer = None
                neuron_arch = None
            
            log("  training model for job({})".format(job_id))
            model = train(train_input_np, train_output_np, algorithm, n_dim, n_classes, n_layer, neuron_arch, k)
            log("  model for job({}) is trained".format(job_id))

            # log("  saving model")
            # save_model(k, job, algorithm, model)
            # log("  model saved")

            log("  testing model for job({})".format(job_id))
            output = test(model, test_input_np, test_output_np, algorithm, n_dim, n_classes, n_layer, neuron_arch, k)
            log("  model for job({}) is tested: {}".format(job_id, output))
            # output_str = json.dumps(output)
            
            log("  updating job({}) status to 'done'".format(job_id))
            update_job_status_to_done(job_id, output)

def data_load_for_k(k):
    log("  loading data for k={}".format(k))
    train_data_df = load_data(k, "train")
    test_data_df = load_data(k, "test")

    log("  loaded data for k={}".format(k))
    n_dim = len(train_data_df.input.loc[0])
    n_classes = len(train_data_df.output.loc[0])

    train_input_np = np.stack(train_data_df.input)
    train_output_np = train_data_df.output_value.to_numpy(dtype=np.int32, copy=True)

    test_input_np = np.stack(test_data_df.input)
    test_output_np = test_data_df.output_value.to_numpy(dtype=np.int32, copy=True)

    del (train_data_df, test_data_df)
    log("  data for k={} is loaded".format(k))
    return n_dim,n_classes,train_input_np,train_output_np,test_input_np,test_output_np

def update_job_status_to_done(job_id, output):
    post_result(job_id, output)
    # con = sqlite3.connect("data/data.db")
    # cur = con.cursor()
    # cur.execute("UPDATE jobs SET status = 'done', when_finished = ?, output_value = ? WHERE id = ?", (datetime.datetime.now(), output, job_id, ))
    # con.commit()
    # con.close()

# def save_model(k, job, algorithm, model):
#     if not os.path.exists("models"):
#         os.makedirs("models")
#     if not os.path.exists("models/k_{}".format(k)):
#         os.makedirs("models/k_{}".format(k))
#     if not os.path.exists("models/k_{}/{}".format(k, algorithm)):
#         os.makedirs("models/k_{}/{}".format(k, algorithm))

#     if algorithm == "mlp":
#         n_layer = job["n_layer"]
#         neuron_arch = job["neuron_arch"]

#         model_name = "model_{}_{}_{}.pkl".format(algorithm, n_layer, neuron_arch)
#     else:
#         model_name = "model_{}.pkl".format(algorithm)
    # pipeline_str = pickle.dumps(model)

    # retry_db(save_model_to_db, job_id, model_name, pipeline_str)

    # dump(model, "models/k_{}/{}/{}.joblib".format(k, algorithm, model_name))

def retry_db(fn, *args, **kwargs):
    while True:
        try:
            return fn(*args, **kwargs)
        except sqlite3.OperationalError:
            log("  sqlite3.OperationalError. Retrying...")
            sleep(1)

# def save_model_to_db(job_id, model_name, pipeline_str):
#     con = sqlite3.connect("data/data.db")
#     cur = con.cursor()
#     cur.execute("INSERT INTO models (job_id, model_str, model_name) VALUES (?, ?, ?)", (job_id, pipeline_str, model_name))
#     con.commit()
#     con.close()

# def update_job_status_to_running(job_id):
#     con = sqlite3.connect("data/data.db")
#     cur = con.cursor()
#     cur.execute("UPDATE jobs SET status = 'running', when_started = ? WHERE id = ?", (datetime.datetime.now(), job_id,))
#     con.commit()
#     con.close()

def fetch_job(k=None):
    return fetch_job_in_queue(k)
    # con = sqlite3.connect("data/data.db")
    # cur = con.cursor()
    # if k:
    #     cur.execute("SELECT id, k, input_value, status, when_enqueued FROM jobs WHERE k=? AND status=? ORDER BY id ASC LIMIT 1", (k, "queued"))
    # else:
    #     cur.execute("SELECT id, k, input_value, status, when_enqueued FROM jobs WHERE status=? ORDER BY id ASC LIMIT 1", ("queued",))
    # row = cur.fetchone()
    # con.close()

    # return row





def pl_simple(classifier, name="classifier"):
  estimators = []
  estimators.append((name, classifier))
  pipeline = Pipeline(estimators)

  return pipeline

def load_data(k, mode):
    if mode not in ("train", "test"):
        raise Exception("unknown mode")

    raw_data = pd.read_csv("data/{}_set_{}.csv".format(mode, k),converters={
        "input": lambda x: list(map(int, x.strip("[]").split(", "))),
        "output": lambda x: list(map(int, x.strip("[]").split(", ")))
        })

    return raw_data

def train(train_input_np, train_output_np, algorithm, n_dim, n_classes, n_layer, neuron_arch, k):
    log("  building the pipeline: {}".format(algorithm))
    if algorithm == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        pipeline = pl_simple(LogisticRegression(solver='liblinear', C=0.05, multi_class='ovr', random_state=0, class_weight='balanced'))
    elif algorithm == "svm":
        from sklearn.svm import SVC
        pipeline =  pl_simple(SVC(gamma='auto', kernel='rbf', class_weight='balanced'))
    elif algorithm == "xgboost":
        pipeline = pl_simple(xgboost.XGBClassifier())
    elif algorithm == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        pipeline = pl_simple(RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0, class_weight='balanced'))
    elif algorithm == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier
        pipeline = pl_simple(DecisionTreeClassifier(max_depth=5, random_state=0, class_weight='balanced'))
    elif algorithm == "mlp":
         # baseline model
        def create_model():
            # create model
            model = Sequential()
            log("  building the nlp model: n_layer={}, neuron_arch={}".format(n_layer, neuron_arch))
            for i in range(n_layer):
                if i == 0:
                    model.add(Dense(neuron_arch[i], input_dim=n_dim, activation='relu'))
                else:
                    model.add(Dense(neuron_arch[i], activation='relu'))

            model.add(Dense(1, activation='sigmoid'))
            # Compile model
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model
        # evaluate model with standardized dataset
        early_stop_10 = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10 )
        estimator = KerasClassifier(model=create_model, epochs=200, batch_size=32, verbose=1, validation_split=0.1, callbacks=[early_stop_10], class_weight='balanced')

        pipeline = pl_simple(estimator)
    else:
        raise Exception("unknown algorithm")
    log("  pipeline build is complete: {}".format(pipeline))
    log("  training model")
    pipeline.fit(train_input_np, train_output_np)
    log("  model trained")

    return pipeline


def test(model, test_input_np, test_output_np, algorithm, n_dim, n_classes, n_layer, neuron_arch, k):
    log("  testing model")
    predictions = model.predict(test_input_np)
    accuracy = accuracy_score(test_output_np, predictions)
    log("  accuracy: algorithm={} k={} accuracy={}".format(algorithm, k, accuracy))
    tn, fp, fn, tp = confusion_matrix(test_output_np, predictions).ravel()
    mcc = matthews_corrcoef(test_output_np, predictions)
    log("  model tested")
    
    return {
        "accuracy": float(accuracy),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "mcc": float(mcc),
        "sensitivity": float(tp / (tp + fn)) if tp + fn > 0 else -1,
        "specificity": float(tn / (tn + fp)) if tn + fp > 0 else -1
    }



import psutil

def get_5_sec_cpu_and_ram_usage():
    cpus = []
    mems = []
    for i in range(5):
        cpus.append(psutil.cpu_percent())
        mems.append(psutil.virtual_memory().percent)
        sleep(1)
    log("  cpu: {}, mem: {}".format(cpus, mems))
    return (sum(cpus) / len(cpus), sum(mems) / len(mems))


work()

# from multiprocessing import Pool

# if __name__ == '__main__':
#     with Pool(10) as p:
#         p.map(work, range(10))

# new_job_id = insert_new_job()
# log("  new job id: {}".format(new_job_id))

# fetched_job = fetch_job_in_queue()
# log("  fetched job: {}".format(fetched_job))

# fetch_result = post_result(fetched_job.get("id"), {"test": 123})
# log("  fetch result: {}".format(fetch_result))

