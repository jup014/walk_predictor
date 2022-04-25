from pandas import read_csv
import pandas as pd
import os
from itertools import product
from datetime import date, timedelta
from .report import log

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
        log("DataLoader.__init__()")
        self.data_path = data_path
        self.model_parameter = model_parameter
        self.raw_data = {}
        
        # to use unlimited memory for large dataframes
        pd.options.mode.chained_assignment = None
        
        self.data = self.load_data()

        

    def load_data(self):
        log("DataLoader.load_data()")
        log("  Loading jawbone.csv data...")
        self.raw_data["jawbone"] = read_csv(os.path.join(self.data_path, 'jawbone.csv'), low_memory=False)
        log("  Loading daily.csv data...")
        self.raw_data["daily"] = read_csv(os.path.join(self.data_path, 'daily.csv'), low_memory=False)
        log("  Loading dose.csv data...")
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
        log("Preprocessor.__init__()")
        self.data_loader = data_loader
        self.model_parameter = data_loader.model_parameter
        self.preprocess()

    def preprocess(self):
        log("Preprocessor.preprocess()")
        if self.model_parameter in ("walk only", "walk and alarm"):
            ###### 1. Selecting columns ######
            log("###### 1. Selecting columns ######")

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

            ## 44 participants in total

            # making a stat of the number of days per user
            stat_user = user_date.groupby(['user'])['local_date'].nunique().sort_values()


            ###### 2. Filter out the participants who have data less than 10 days ######
            log("###### 2. Filter out the participants who have data less than 10 days ######")

            # filter out users that have less days of data than THRESHOLD_OF_DAYS_PER_USER
            users_to_be_removed = stat_user[stat_user < Preprocessor.THRESHOLD_OF_DAYS_PER_USER].index

            # three participants were removed

            log("Threshold: {}".format(Preprocessor.THRESHOLD_OF_DAYS_PER_USER))
            log("Users to be removed:{}".format(list(users_to_be_removed)))

            jawbone4 = jawbone3[~jawbone3["user"].isin(users_to_be_removed)]

            user_date2 = user_date[~user_date["user"].isin(users_to_be_removed)]


            ###### 3. Generate basic data stats ######
            log("###### 3. Generate basic data stats ######")

            log("Number of user-minute pairs: {}".format(len(user_date2)))
            log("Number of minutes per user: ")
            user_date2_stat = user_date2.groupby(['user'])['local_date'].nunique().sort_values()
            # log(user_date2_stat)    ## too long
            
            # Mean: 43.26829268292683 Stdev: 9.088521305041603
            log("  Mean: {} Stdev: {}".format(user_date2_stat.mean(), user_date2_stat.std()))
            
            # printing the amount of data removed
            jawbone3_count, _ = jawbone3.shape
            jawbone4_count, _ = jawbone4.shape

            log("Shape Change: {} -> {} (-{}, -{}%)".format(
                jawbone3_count, 
                jawbone4_count, 
                jawbone3_count - jawbone4_count, 
                round((jawbone3_count - jawbone4_count) / jawbone3_count * 100, 2)
                )
            )

            ######
            all_minutes = jawbone4.groupby(['user']).start_utime_local.nunique().sort_values()
            over60 = jawbone4[jawbone4["steps"] > Preprocessor.MINIMUM_STEPS_PER_MINUTE].groupby(['user']).start_utime_local.nunique().sort_values()
            log("all: {}(+_{}) over60: {}(+_{})".format(all_minutes.mean(), all_minutes.std(), over60.mean(), over60.std()))

            minute_stat = pd.merge(user_date2_stat, all_minutes, left_index=True, right_index=True, how="left").merge(over60, left_index=True, right_index=True, how="left")
            minute_stat['all_per_day'] = minute_stat['start_utime_local_x'] / minute_stat['local_date']
            minute_stat['over60_per_day'] = minute_stat['start_utime_local_y'] / minute_stat['local_date']
            log("per day: all: {}(+_{}) over60: {}(+_{})".format(minute_stat['all_per_day'].mean(), minute_stat['all_per_day'].std(), minute_stat['over60_per_day'].mean(), minute_stat['over60_per_day'].std()))

            ###############################################################################

            # adding cumulative minute index

            jawbone5 = jawbone4.join(jawbone4.groupby(["user"])['local_date'].agg(['min']), on='user')
            jawbone5["cumulative_minute_index"] = int((jawbone5["local_date"] - jawbone5["min"])[500].total_seconds()/60) + jawbone5["local_minute_index"]

            merged_walks = self.get_merged_walks(jawbone5)



            ###############################################################################
            # prepare the data for the walk calculation
            current_vector = jawbone5[["user", "start_utime_local", "local_date", "local_minute_index", "steps", "cumulative_minute_index"]]
            current_vector["add_count"] = 1
            current_vector = current_vector[current_vector["steps"] > Preprocessor.MINIMUM_STEPS_PER_MINUTE]
            current_vector = current_vector[["user", "local_date", "start_utime_local", "local_minute_index", "add_count", "cumulative_minute_index"]]

            # iteratively calculate the walk
            vector_history = []
            
            for i in range(0, Preprocessor.MINIMUM_NUMBER_OF_MINUTES_FOR_A_WALK):
                log("Iteration: {}, length: {}".format(i, current_vector.shape[0]))
                new_vector = calculate_walk(current_vector)
                current_vector = new_vector

            log("Final, length: {}".format(current_vector.shape[0]))

            consecutive_minutes = current_vector[["user", "local_date", "local_minute_index"]].drop_duplicates()


            all_walks = current_vector.groupby(['user']).cumulative_minute_index.nunique().sort_values()
            log("all_walks: {}(+_{})".format(all_walks.mean(), all_walks.std()))

            all_walk_stat = pd.merge(user_date2_stat, all_walks, left_index=True, right_index=True, how="left")
            minute_stat['all_per_day'] = minute_stat['cumulative_minute_index_x'] / minute_stat['local_date']
            minute_stat['over60_per_day'] = minute_stat['cumulative_minute_index_y'] / minute_stat['local_date']
            log("per day: all: {}(+_{}) over60: {}(+_{})".format(minute_stat['all_per_day'].mean(), minute_stat['all_per_day'].std(), minute_stat['over60_per_day'].mean(), minute_stat['over60_per_day'].std()))

    def append_new_row(self, cur_user, start_cumulative_minute_index, cur_cumulative_minute_index, cur_local_date, merged_walks):
        merged_walks = pd.concat([merged_walks,
                            pd.DataFrame(data={
                                "user": [cur_user],
                                "local_date": [cur_local_date],
                                "start_day": [start_cumulative_minute_index % 1440],
                                "end_day": [cur_cumulative_minute_index % 1440],
                                "start_cum": [start_cumulative_minute_index],
                                "end_cum": [cur_cumulative_minute_index],
                                "length": [cur_cumulative_minute_index - start_cumulative_minute_index + 1]
                            })])
            
        return merged_walks


    def get_merged_walks(self, jawbone5):
        save_file_path = "./data/merged_walks.csv"
        if os.path.exists(save_file_path):
            merged_walks = pd.read_csv(save_file_path)
        else:
            sorted_minutes = jawbone5[jawbone5['steps']>Preprocessor.MINIMUM_STEPS_PER_MINUTE].sort_values(by=["user", "cumulative_minute_index"])
            sorted_minutes.reset_index()
            cur_user = None
            start_cumulative_minute_index = None
            cur_cumulative_minute_index = None
            cur_local_date = None
            merged_walks = pd.DataFrame(data={
                "user": [],
                "local_date": [],
                "start_day": [],
                "end_day": [],
                "start_cum": [],
                "end_cum": [],
                "length": []
            })

            for index, row in sorted_minutes.iterrows():
                # log("{} {} {}".format(row.user, row.cumulative_minute_index, row.steps))
                # if index % 1000 == 0:
                # log("{}/{}".format(index, sorted_minutes.shape[0]))
                if row.user != cur_user:
                    log("user: {}".format(row.user))
                    if cur_cumulative_minute_index:
                        merged_walks = self.append_new_row(cur_user, start_cumulative_minute_index, cur_cumulative_minute_index, cur_local_date, merged_walks)
                    cur_user = row.user
                    start_cumulative_minute_index = row.cumulative_minute_index
                    cur_cumulative_minute_index = row.cumulative_minute_index
                    cur_local_date = row.local_date
                else:
                    if row.cumulative_minute_index != cur_cumulative_minute_index + 1:
                        # log("  new cum min index: {}".format(row.cumulative_minute_index))
                        merged_walks = self.append_new_row(cur_user, start_cumulative_minute_index, cur_cumulative_minute_index, cur_local_date, merged_walks)
                        start_cumulative_minute_index = row.cumulative_minute_index
                        cur_cumulative_minute_index = row.cumulative_minute_index
                        cur_local_date = row.local_date
                    else:
                        cur_cumulative_minute_index = row.cumulative_minute_index
            if cur_user:
                if cur_cumulative_minute_index:
                    merged_walks = self.append_new_row(cur_user, start_cumulative_minute_index, cur_cumulative_minute_index, cur_local_date, merged_walks)
            merged_walks.to_csv(save_file_path, index=False)
        return merged_walks
        