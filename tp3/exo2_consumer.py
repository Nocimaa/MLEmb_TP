from kafka import KafkaConsumer
import json
import numpy as np


producer = KafkaConsumer(bootstrap_servers=["nowledgeable.com:9092"], auto_offset_reset="earliest")
producer.subscribe(["am-exo2"])

for msg in producer:
    dic = json.loads(msg.value)
    arr = np.array(dic["data"])
    print(arr)