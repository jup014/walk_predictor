import os
import pymongo
from lib.db import get_mongo_client, get_stat, rollback_all_running
from lib.report import log


# client = get_mongo_client()

# rollback_all_running(client)

# log(get_stat(['status'], client))

conn_str = "mongodb+srv://new-user-001:{}@cluster0.xdph2.mongodb.net/Cluster0?retryWrites=true&w=majority".format(os.environ["MONGO_PASSWORD"])
client = pymongo.MongoClient(conn_str)

print(list(client['sci-writing']['jobs'].aggregate(
    [
        {
            "$group": 
            {
                "_id": "$status",
                "count": { "$sum": 1 }
            }
        }
    ])))