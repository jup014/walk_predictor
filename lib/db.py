
import datetime
import os
import pymongo

def get_mongo_client():
    conn_str = "mongodb+srv://new-user-001:{}@cluster0.xdph2.mongodb.net/Cluster0?retryWrites=true&w=majority".format(os.environ["MONGO_PASSWORD"])
    client = pymongo.MongoClient(conn_str)
    
    return client['sci-writing']['jobs']

def build_new_job(params={}):
    params['when_created'] = datetime.datetime.now()
    params['status'] = "queued"
    params['when_started'] = None
    params['when_finished'] = None
    params['output'] = None

    return params
def insert_new_job(params={}, client=None):
    params = build_new_job(params)

    if client is None:
        client = get_mongo_client()
    return client.insert_one(params).inserted_id

def insert_many_new_jobs(list_of_params, client=None):
    if client is None:
        client = get_mongo_client()
    return client.insert_many(list_of_params)

def fetch_job_in_queue(k=None, client=None):
    if client is None:
        client = get_mongo_client()
    if k is None:
        return client.find_one_and_update(
            {
                "status": "queued"
            },
            {
                "$set": {
                    "status": "running",
                    "when_started": datetime.datetime.now()
                }
            },
            return_document=pymongo.ReturnDocument.AFTER
        )
    else:
        return client.find_one_and_update(
            {
                "status": "queued",
                "k": k
            },
            {
                "$set": {
                    "status": "running",
                    "when_started": datetime.datetime.now()
                }
            },
            return_document=pymongo.ReturnDocument.AFTER
        )

def post_result(job_id, output, client=None):
    if client is None:
        client = get_mongo_client()
    return client.find_one_and_update(
        {
            "_id": job_id
        },
        {
            "$set": {
                "status": "finished",
                "when_finished": datetime.datetime.now(),
                "output": output
            }
        },
        return_document=pymongo.ReturnDocument.AFTER
    )
