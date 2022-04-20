from pandas import read_csv
import pandas as pd
import os
from itertools import product
from datetime import date, timedelta

def get_date(x):
    return date(x.year, x.month, x.day)

# convert a datetime object to an integer, which denotes the number of minutes since midnight
def get_minute_index(x):
    return (x.hour * 60) + x.minute


# return a range of dates
def date_range(start_date, end_date):
    delta = end_date - start_date

    for i in range(delta.days + 1):
        yield start_date + timedelta(days=i)

# define an iterative walk calculation (merging consecutive active minutes)
def calculate_walk(cv):
    nv = cv.copy(deep=True)
    nv["prev_minute_index"] = nv["cumulative_minute_index"] - 1
    
    nv = nv[["user", "local_date", "prev_minute_index"]]
    jv = cv.merge(nv, left_on=["user", "local_date", "cumulative_minute_index"], right_on=["user", "local_date", "prev_minute_index"], how="inner")
    jv["add_count"] += 1
    jv = jv[["user", "local_date", "local_minute_index", "cumulative_minute_index", "add_count"]]

    return jv 

# generate complete product of vectors
def product_df(mat1, mat2):
    mat1 = mat1.drop_duplicates()
    mat2 = mat2.drop_duplicates()

    temp = pd.DataFrame(list(product(mat1.values, mat2.values)))
    for i, acol in enumerate(mat1.columns):
        temp[acol] = temp[0].apply(lambda x: x[i])
    for i, acol in enumerate(mat2.columns):
        temp[acol] = temp[1].apply(lambda x: x[i])
    temp = temp.drop(columns=[0, 1])
    return temp


class DataLoader:
    def __init__(self, data_path, model_parameter):
        self.data_path = data_path
        self.model_parameter = model_parameter
        self.raw_data = {}
        
        # to use unlimited memory for large dataframes
        pd.options.mode.chained_assignment = None
        
        self.data = self.load_data()

        

    def load_data(self):
        self.raw_data["jawbone"] = read_csv(os.path.join(self.data_path, 'jawbone.csv'), low_memory=False)
        self.raw_data["daily"] = read_csv(os.path.join(self.data_path, 'daily.csv'), low_memory=False)
        self.raw_data["dose"] = read_csv(os.path.join(self.data_path, 'dose.csv'), low_memory=False)
        

class Preprocessor:

    # cut off values that are not in the range of the data
    THRESHOLD_OF_DAYS_PER_USER = 10

    # cut off values for the number of consecutive minutes for a walk
    MINIMUM_NUMBER_OF_MINUTES_FOR_A_WALK = 5

    # cut off values for the number of steps per minute for an active minute
    MINIMUM_STEPS_PER_MINUTE = 60

    # cut off value for the number of weeks for looking back
    NUMBER_OF_WEEKS_FOR_LOOKING_BACK = 5


    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.model_parameter = data_loader.model_parameter
        self.preprocess()

    def preprocess(self):
        if self.model_parameter in ("walk only", "walk and alarm"):
            # Column names of jawbone data
            # 'Var1', 'user', 'start_datetime', 'end_datetime', 'timezone', 'userid',
            # 'steps', 'gmtoff', 'tz', 'start_date', 'end_date', 'start_utime',
            # 'end_utime', 'start_udate', 'end_udate', 'intake_date', 'intake_utime',
            # 'intake_tz', 'intake_gmtoff', 'intake_hour', 'intake_min',
            # 'intake_slot', 'travel_start', 'travel_end', 'exit_date',
            # 'dropout_date', 'last_date', 'last_utime', 'last_tz', 'last_gmtoff',
            # 'last_hour', 'last_min', 'start_utime_local', 'end_utime_local'


            # duplicate jawbone data
            jawbone2 = self.data_loader.raw_data["jawbone"].copy(deep=True)

            # selecting only important columns
            jawbone3 = jawbone2[["user", "start_utime_local", "steps"]]

            # convert string date fields to datetime objects
            jawbone3["start_utime_local"] = pd.to_datetime(jawbone3["start_utime_local"])

            # picking up the local date
            jawbone3["local_date"] = jawbone3["start_utime_local"].apply(get_date)

            # picking up the local minute index
            jawbone3["local_minute_index"] = jawbone3["start_utime_local"].apply(
                get_minute_index)

            # picking up the user - date data
            user_date = jawbone3[["user", "local_date"]].drop_duplicates()

            # making a stat of the number of days per user
            stat_user = user_date.groupby(['user'])['local_date'].nunique().sort_values()


            # filter out users that have less days of data than THRESHOLD_OF_DAYS_PER_USER
            users_to_be_removed = stat_user[stat_user < Preprocessor.THRESHOLD_OF_DAYS_PER_USER].index

            print("Threshold: {}".format(Preprocessor.THRESHOLD_OF_DAYS_PER_USER))
            print("Users to be removed:{}".format(list(users_to_be_removed)))

            jawbone4 = jawbone3[~jawbone3["user"].isin(users_to_be_removed)]

            # user_date2 = user_date[~user_date["user"].isin(users_to_be_removed)]

            # printing the amount of data removed
            jawbone3_count, _ = jawbone3.shape
            jawbone4_count, _ = jawbone4.shape

            print("Shape Change: {} -> {} (-{}, -{}%)".format(
                jawbone3_count, 
                jawbone4_count, 
                jawbone3_count - jawbone4_count, 
                round((jawbone3_count - jawbone4_count) / jawbone3_count * 100, 2)
                )
            )

            ###############################################################################

            # adding cumulative minute index

            jawbone5 = jawbone4.join(jawbone4.groupby(["user"])['local_date'].agg(['min']), on='user')
            jawbone5["cumulative_minute_index"] = int((jawbone5["local_date"] - jawbone5["min"])[500].total_seconds()/60) + jawbone5["local_minute_index"]


            ###############################################################################
            # prepare the data for the walk calculation
            current_vector = jawbone5[["user", "local_date", "local_minute_index", "steps", "cumulative_minute_index"]]
            current_vector["add_count"] = 1
            current_vector = current_vector[current_vector["steps"] > Preprocessor.MINIMUM_STEPS_PER_MINUTE]
            current_vector = current_vector[["user", "local_date", "local_minute_index", "add_count", "cumulative_minute_index"]]

            # iteratively calculate the walk
            vector_history = []
            
            for i in range(0, Preprocessor.MINIMUM_NUMBER_OF_MINUTES_FOR_A_WALK):
                print("Iteration: {}, length: {}".format(i, current_vector.shape[0]))
                new_vector = calculate_walk(current_vector)
                current_vector = new_vector

            print("Final, length: {}".format(current_vector.shape[0]))

            consecutive_minutes = current_vector[["user", "local_date", "local_minute_index"]].drop_duplicates()