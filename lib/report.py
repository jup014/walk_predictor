import datetime
import os
import threading


def log(msg=""):
    f_msg = "{}\t{}\t{}\t{}".format(datetime.datetime.now(), os.getpid(), threading.get_ident(), msg)

    print(f_msg)
    with open("log.txt", "a") as f:
        f.write(f_msg + "\n")

class Reporter:
    def __init__(self, tester):
        self.tester = tester
    
    def report(self, data_path):
        pass