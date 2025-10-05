from kafka import KafkaProducer
import json

producer = KafkaProducer(bootstrap_servers=["nowledgeable.com:9092"])

producer.send("am-exo2", json.dumps({"data" : [[1,2], [3,4]]}).encode())


producer.flush()
producer.close()