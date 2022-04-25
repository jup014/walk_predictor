import traceback
from lib.preprocess import DataLoader, Preprocessor
from lib.train_test import Trainer, Tester
from lib.report import Reporter, log

data_path = 'data/'

model_parameter = "walk only" # "walk only" or "walk and alarm"

try:
    dl = DataLoader(data_path, model_parameter)
    pp = Preprocessor(dl)

    trainer = Trainer(pp)
    tester = Tester(trainer)

    reporter = Reporter(tester)
    reporter.report(data_path)
except Exception as e:
    tb = traceback.format_exc()
    log(tb)