from flask import Flask, request
from gevent.pywsgi import WSGIServer
from geventwebsocket.handler import WebSocketHandler
import json
# import cv2threading
from PIL import Image
import requests
from io import BytesIO
import base64

from concurrent.futures import ThreadPoolExecutor
import time

app = Flask(__name__)

executor = ThreadPoolExecutor(10)
dizhi="/v1/object-detection"
# 模拟耗时任务
def run_job(encode,url,start_time,carnum):
    try:
        producer = cv2threading.Producer(url, encode, start_time,carnum)
        producer.start()
        print('run_job complete')
    except Exception as e:
        print(e)
        pass
@app.route(dizhi+'/pic/', methods=['POST'])
def detection():
    if request.method == 'POST':
        now=time.time()
        data = request.get_data()
        json_data = json.loads(data)
        service = json_data['service']
        # print(service)
        if service =='picbase':
            base = json_data['image']
            image_str = BytesIO(base64.b64decode(base))
            imgData = Image.open(image_str)
        elif service == 'picurl':
            url = json_data['image']
            # print(url)
            response = requests.get(url)
            # print(response)
            imgData = Image.open(BytesIO(response.content))
            # imgData = Image.open(image_str)
        else:
            return {"success":False}
        res, returndict = yoloclass.detect_image(imgData)  # 算法检测接口
        # print('花费了'+str(time.time()-now)+'秒的时间')
        if returndict:
            j = json.dumps(obj=returndict)
            sample = [{"class": "puromisu", "score": "0.8895045", "h": 86, "w": 50, "x": 50, "y": 50},
                      {"class": "puromisu2", "score": "0.85214925", "h": 86, "w": 50, "x": 50, "y": 50}]
            # print(j)
            return j
        else:
            return 'null'
    else:
        return 'null'
@app.route(dizhi+'/videotape/', methods=['POST'])
def detection_video():
    data = request.get_data()
    json_data = json.loads(data)
    url = json_data['video']
    encoded = json_data['encoded']
    start_time = 0
    carnum=""
    try:
        carnum = json_data['carnum']
        start_time = json_data['start_time']
    except:
        pass
    c = 1
    executor.submit(run_job, carnum=carnum,encode=encoded, url=url,start_time=start_time)
    print(url, encoded, start_time,carnum)
    return 'ok'

if __name__ == "__main__":
    http_server = WSGIServer(('0.0.0.0', 10006), app,handler_class=WebSocketHandler)
    print('Listening on address: http//127.0.0.1')
    http_server.serve_forever()