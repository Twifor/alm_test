import os
import json


def logger(event_id, obj):
    log_path = "./logs/"
    if os.path.exists(log_path) == False:
        os.mkdir(log_path)
    event_path = log_path + event_id + ".log"
    if os.path.exists(event_path) == True:
        raise ValueError(
            event_path + "already exists. Please modify your event_id.")
    simple_log = open(event_path, "w")
    json_str = json.dumps(obj, indent=4, separators=(',', ':'))
    simple_log.write(json_str)
    simple_log.close()
