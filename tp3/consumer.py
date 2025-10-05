from kafka import KafkaConsumer





consumer = KafkaConsumer(bootstrap_servers=["nowledgeable.com:9092"], auto_offset_reset="earliest")

# consumer.subscribe(["exo1"])
consumer.subscribe(["am-exo2"])

for msg in consumer:
    print(msg)