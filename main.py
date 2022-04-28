import traceback
from lib.preprocess import DataLoader, Preprocessor
from lib.train_test import Organizer, Tester
from lib.report import Reporter, log

data_path = 'data/'

model_parameter = "walk only" # "walk only" or "walk and alarm"

print("hello")

try:
    dl = DataLoader(data_path, model_parameter)
    pp = Preprocessor(dl)

    trainer = Organizer(pp)
except Exception as e:
    tb = traceback.format_exc()
    log(tb)

