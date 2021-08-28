import minio
from datetime import datetime
from kafka import KafkaProducer
import json
minio_conf = {
    'endpoint': '192.168.1.128:9000',
    'access_key': 'admin',
    'secret_key': '12345678',
    'secure': False
}
def up_data_minio(bucket: str,filepath,objectname):
    client = minio.Minio(**minio_conf)
    client.fput_object(bucket_name=bucket, object_name=objectname,
                       file_path=filepath,
                       content_type='application/jpg'
                       )

def send_topic_msg(json_data,bootserver,topic):
    # kafkaClient = KafkaProducer(security_protocol="SSL",bootstrap_servers=['192.168.11.107:40711','192.168.11.114:41411','192.168.11.115:41511'],api_version=(0,11,5),
    #           value_serializer=lambda x: json.dumps(x).encode('utf-8'))
    producer = KafkaProducer(
        # security_protocol="SSL",
        # api_version = (0, 11, 5),
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        bootstrap_servers=[bootserver]
    )
    # json_data = {"openId": "", "dappAppId": "", "caCrt": str(base64_file, 'utf-8'),
    #              "nodeKey": "", "nodeCrt": ""}
    producer.send(topic, json_data)
    print(json_data)
    producer.close()