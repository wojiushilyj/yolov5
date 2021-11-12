import minio
from datetime import datetime
from kafka import KafkaProducer
import json
import cv2
import detect_new
import uuid
import urllib.parse
import time
# minio_conf = {
#     'endpoint': '192.168.1.128:9000',
#     'access_key': 'admin',
#     'secret_key': '12345678',
#     'secure': False
# }
#政务云
minio_conf = {
    'endpoint': '10.18.211.186:9000',
    'access_key': 'reload',
    'secret_key': 'Abc12345678',
    'secure': False
}
tem_pic=R"D:\python\2021828yolov5\yolov5\image/"
minio_root="http://10.18.211.186:9000/screenshot/"
cdh_url="cdh-1:40711"
master_url="master:9092"
#minio
def up_data_minio(bucket: str,filepath,objectname):
    client = minio.Minio(**minio_conf)
    client.fput_object(bucket_name=bucket, object_name=objectname,
                       file_path=filepath,
                       content_type='application/jpg'
                       )

#卡夫卡
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
#播放url
class playurl(object):
    def __init__(self, singleurl,encoded='',start_time='',carnum=''):
        self.singleurl=singleurl
        self.a = detect_new.detect_api()
        self.encoded = encoded
        self.start_time = start_time
        self.carnum = carnum
    def urlparseresult(self):
        result = urllib.parse.urlsplit(self.singleurl)
        query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(self.singleurl).query))
        if query['cameraid']:
            return query['cameraid']
        else:
            return None
    def playon(self):
        if not isinstance(self.singleurl,str):
            return
        cap = cv2.VideoCapture(self.singleurl)
        c = 0
        shotsavetime = '0'
        while (cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                print("Opening camera is failed")
                break
            # 时间差1秒截图
            gaptime = int(datetime.now().now().strftime('%Y%m%d%H%M%S')) - int(shotsavetime)
            result, names = self.a.detect([frame])
            if result[0][1]:
                car_type = ''
                for i in result[0][1]:
                    if i[0] in (0, 1, 2, 3):
                        car_type = names[i[0]]
                for i in result[0][1]:
                    if (i[0] in [6,7,8]) and i[2]>0.8 and gaptime>3:
                        cv2.imwrite(tem_pic + str(c) + '.jpg', result[0][0])
                        shotsavetime = datetime.now().now().strftime('%Y%m%d%H%M%S')
                        uuid_str = uuid.uuid4().hex
                        save_screenshot = datetime.now().now().strftime('%Y%m%d%H%M%S') + '_%s.jpg' % uuid_str
                        up_data_minio('screenshot',
                                                       tem_pic + str(c) + '.jpg',
                                                       save_screenshot)
                        c = c + 1
                        kafkajson = {'type': names[i[0]],
                                     'pic_path': minio_root + save_screenshot,
                                     'encoded': self.urlparseresult,"car_type":car_type,"screenshot_time":time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}
                        print(kafkajson)
                        # 推送识别信息
                        send_topic_msg(kafkajson, cdh_url, 'video-livestreaming-recognition')
                        send_topic_msg(kafkajson, master_url, 'video-livestreaming-recognition')
    def playvideorecord(self):
        if not isinstance(self.singleurl,str):
            return
        cap = cv2.VideoCapture(self.singleurl)
        c = 0
        shotsavetime = '0'
        fps = cap.get(cv2.CAP_PROP_FPS)  # 视频的帧率FPS
        total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 视频的总帧数
        # while (cap.isOpened()):
        for i in range(int(total_frame)):
            # ret, frame = cap.read()
            ret = cap.grab()
            if not ret:
                print("Opening camera is failed")
                break
            if i%fps ==0:
                #视频检测到第几秒
                stime=i//fps
                ret1, frame = cap.retrieve()
                if ret1:
                    # 时间差1秒截图
                    gaptime = int(datetime.now().now().strftime('%Y%m%d%H%M%S')) - int(shotsavetime)
                    result, names = self.a.detect([frame])
                    if result[0][1]:
                        car_type=''
                        car_license=''
                        car_color=''
                        for i in result[0][1]:
                            if i[0] in (0,1,2,3):
                                from PIL import Image
                                from hsv_color.color_detect import get_color
                                import numpy as np
                                car_type = names[i[0]]
                                if names[i[0]]=='jiaobanche':
                                    car_color = "white"
                                else:
                                    detect_car_image=Image.fromarray(result[0][2])
                                    vas=detect_car_image.crop(i[1])
                                    vas = cv2.cvtColor(np.asarray(vas), cv2.COLOR_RGB2BGR)
                                    vehicle_color = get_color(vas)
                                    car_color=vehicle_color
                            if i[0] in (4,5):
                                car_license=names[i[0]]
                        for i in result[0][1]:
                            if (i[0] in [6,7,8]) and i[2]>0.8 and gaptime>10:
                                print(tem_pic + 'videorecord'+str(c) + '.jpg')
                                cv2.imwrite(tem_pic + 'videorecord'+str(c) + '.jpg', result[0][0])
                                cv2.imwrite(tem_pic + 'videorecord2_' + str(c) + '.jpg', result[0][2])
                                shotsavetime = datetime.now().now().strftime('%Y%m%d%H%M%S')
                                uuid_str = uuid.uuid4().hex
                                save_screenshot = shotsavetime + '_%s.jpg' % uuid_str
                                # uuid_str1 = uuid.uuid4().hex
                                save_screenshot2 = 'origin_'+shotsavetime + '_%s.jpg' % uuid_str
                                up_data_minio('screenshot',
                                               tem_pic +'videorecord'+ str(c) + '.jpg',
                                                save_screenshot)
                                up_data_minio('screenshot',
                                              tem_pic + 'videorecord2_' + str(c) + '.jpg',
                                              save_screenshot2)
                                try:
                                    timeArray = time.localtime(int(self.start_time)+stime)
                                    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
                                except:
                                    otherStyleTime = ''

                                c = c + 1
                                #车辆分辨颜色
                                from hsv_color.color_detect import get_color
                                vehicle_color=get_color(result[0][2].crop(1,1,1,1))
                                print(vehicle_color)
                                # kafkajson = {'type': names[i[0]],
                                #              'pic_path': minio_root + save_screenshot,
                                #              'original_pic_path': minio_root + save_screenshot2,
                                #              'encoded': self.encoded,'carnum':self.carnum,'screenshot_time': otherStyleTime,'car_type':car_type,'car_license':car_license}
                                kafkajson = {'type': names[i[0]],
                                             'pic_path': minio_root + save_screenshot,
                                             'original_pic_path': minio_root + save_screenshot2,
                                             'encoded': self.encoded,
                                             'screenshot_time': otherStyleTime,
                                             'car_type': car_type,
                                             'car_license': car_license,
                                             'carnum':self.carnum,
                                             'car_color':car_color
                                             }
                                print(kafkajson)
                                # 推送识别信息
                                send_topic_msg(kafkajson, cdh_url, 'video-recognition-1')
                                send_topic_msg(kafkajson, master_url, 'video-recognition')
                else:
                    print("Error retrieving frame from movie!")
                    break