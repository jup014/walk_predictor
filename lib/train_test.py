import datetime
import json
import os
import numpy as np
import pandas as pd
import sqlite3

from lib.report import log


class Organizer:
    K = 10
    N_LAYER_MAX = 10
    N_NEURON_MIN = 10
    N_NEURON_MAX = 5000
    N_EXPERIMENT = 10

    def __init__(self, preprocessor):
        log()
        log("Organizer: start")
        self.preprocessor = preprocessor
        self.data_for_model = self.preprocessor.data_for_model

        log("Organizer: balance data")
        self.balance_data()
        log("Organizer: split data")
        self.split_data()
        log("Organizer: create db")
        self.create_db()
        log("Organizer: enqueue")
        self.enqueue()
        log("Organizer: end")
    
    def create_db(self):
        con = sqlite3.connect("data/data.db")

        cur = con.cursor()

        cur.execute("CREATE TABLE IF NOT EXISTS jobs (id INTEGER PRIMARY KEY, k INT, input_value TEXT, status TEXT, when_enqueued DATETIME, when_started DATETIME, when_finished DATETIME, output_value TEXT)")
        # cur.execute("CREATE TABLE IF NOT EXISTS models (id INTEGER PRIMARY KEY, job_id INT, model_str TEXT, model_name TEXT)")
        
        con.commit()
        con.close()

    def enqueue(self):
        con = sqlite3.connect("data/data.db")

        cur = con.cursor()

        algorithms = ["logistic_regression", "svm", "xgboost", "random_forest", "decision_tree"]

        for algorithm in algorithms:
            for i in range(self.K):
                param = {
                    "k": i,
                    "algorithm": algorithm,
                }
                cur.execute("INSERT INTO jobs (input_value, k, status, when_enqueued) VALUES (?, ?, ?, ?)", (json.dumps(param), i, "queued", datetime.datetime.now()))

        for experiment_number in range(self.N_EXPERIMENT):
            for n_layer in range(1, Organizer.N_LAYER_MAX):
                neuron_arch = np.random.randint(Organizer.N_NEURON_MIN, Organizer.N_NEURON_MAX, size=n_layer).tolist()
                
                for i in range(self.K):
                    param = {
                        "k": i,
                        "algorithm": "mlp",
                        "n_layer": n_layer,
                        "neuron_arch": neuron_arch,
                        "experiment_number": "{}_{}".format(experiment_number, n_layer)
                    }
                    cur.execute("INSERT INTO jobs (input_value, k, status, when_enqueued) VALUES (?, ?, ?, ?)", (json.dumps(param), i, "queued", datetime.datetime.now()))
        con.commit()
        con.close()

    def split_data(self):
        total_count = self.data_for_model.shape[0]

        self.data_for_model = self.data_for_model.sample(n=total_count, replace=False, random_state=42).reset_index(drop=True)

        if os.path.exists("data/test_set_{}.csv".format(Organizer.K - 1)):
            pass
        else:
            for i in range(self.K):
                train_set = pd.concat([self.data_for_model[:i*(total_count//self.K)], self.data_for_model[(i+1)*(total_count//self.K):]])
                test_set = self.data_for_model[i*(total_count//self.K):(i+1)*(total_count//self.K)]
                log("    exporting train and test set for k={}".format(i))
                train_set.to_csv(f"data/train_set_{i}.csv", index=False)
                test_set.to_csv(f"data/test_set_{i}.csv", index=False)

    def balance_data(self):
        df = self.data_for_model
        pos_set = df[df.output_value==1]
        neg_set = df[df.output_value==0]
        pos_count = pos_set.shape[0]
        neg_count = neg_set.shape[0]

        if pos_count > neg_count:
            pos_set = pos_set.sample(n=neg_count, replace=False, random_state=42)
        elif pos_count < neg_count:
            neg_set = neg_set.sample(n=pos_count, replace=False, random_state=42)
        total_set = pd.concat([pos_set, neg_set])

        self.data_for_model = total_set
        
        

class Tester:
    def __init__(self, trainer):
        self.trainer = trainer