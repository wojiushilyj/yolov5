from flask import Flask, render_template, Response,request
import cv2
import detect_new
import json
from concurrent.futures import ThreadPoolExecutor
import play_kafka_minio
import time
from io import BytesIO
import requests
import base64
from PIL import Image
import numpy
import argparse
app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=20)
executor2 = ThreadPoolExecutor(max_workers=20)
DETECTION_URL = "/v1/object-detection"
a = detect_new.detect_api()
# 模拟耗时任务
def run_job(url):
    try:
        p=play_kafka_minio.playurl(singleurl=url)
        p.playon()
        print('background complete')
    except Exception as e:
        print(e)
        pass

# 录像识别耗时任务2
def run_job2(url,carnum, encoded, start_time):
    # try:
    p1=play_kafka_minio.playurl(singleurl=url,encoded=encoded,carnum=carnum,start_time=start_time)
    p1.playvideorecord()
    print('record video complete')
    # except Exception as e:
    #     print(e)
    #     pass
def gen_frames(params):
    print(params)
    camera = cv2.VideoCapture(params)
    while True:
        # 一帧帧循环读取摄像头的数据
        success, frame = camera.read()
        if not success:
            break
        else:
            result, names = a.detect([frame])
            # cv2.imwrite('1.jpg', result[0][0])
            # 将每一帧的数据进行编码压缩，存放在memory中
            ret, buffer = cv2.imencode('.jpg', result[0][0])
            frame = buffer.tobytes()
            # 使用yield语句，将帧数据作为响应体返回，content-type为image/jpeg
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#
def return_img_stream1(params):
    print(params)
    camera = cv2.VideoCapture(params)
    while True:
        # 一帧帧循环读取摄像头的数据
        success, frame = camera.read()
        if not success:
            break
        else:
            result, names = a.detect([frame])
            # cv2.imwrite('1.jpg', result[0][0])
            # 将每一帧的数据进行编码压缩，存放在memory中
            ret, buffer = cv2.imencode('.jpg', result[0][0])
            image = cv2.imencode('.jpg', buffer)[1]
            frame = str(base64.b64encode(image))[2:-1]
            # frame = buffer.tobytes()
            # 使用yield语句，将帧数据作为响应体返回，content-type为image/jpeg
            return (frame)
def bendishibie(image):
    import json, base64
    import requests
    from io import BytesIO
    try:
        headers = {"Content-type": "application/json"}
        url = 'http://localhost:12909/predict/ocr_system'
        # img_file = '11.jpg'
        # with open(img_file, 'rb') as f:
        #     str1 = base64.b64encode(f.read())
        #     str1 = str1.decode("utf-8")
        output_buffer = BytesIO()
        image.save(output_buffer, format='PNG')
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data)
        data = {'images': [str(base64_str, encoding = "utf-8")]}
        # print(data)
        r = requests.post(url=url,headers=headers,data=json.dumps(data)).json()
        # print(r,type(r))
        # r1 = eval(r.content)
        res = r['results'][0]
        str3 = ''
        for content in res:
            str3 += str(content['text']) + '\n'
        return str3.replace('\n','')
    except:
        return '0'


#直播流识别返回图片流
@app.route(DETECTION_URL+'/video_start',  methods=['GET'])
def video_start():
    params = request.args.get('params')
    params = params.replace( "'", "" ).replace( "\"", "" )
    # 通过将一帧帧的图像返回，就达到了看视频的目的。multipart/x-mixed-replace是单次的http请求-响应模式，如果网络中断，会导致视频流异常终止，必须重新连接才能恢复
    return Response(gen_frames(params), mimetype='multipart/x-mixed-replace; boundary=frame')
#直播流页面
@app.route(DETECTION_URL+'/video_page')
def video_page():
    params = request.args.get('params')
    params = params.replace("'", "").replace("\"", "")
    return render_template('video_start.html',params=params)

#后台直播流加载
@app.route(DETECTION_URL+'/backthread',methods=['POST'])
def backthread():
    data = request.get_data()
    json_data = json.loads(data)
    url = json_data['url']
    for i in url:
        executor.submit(run_job, url=i)
    return "ok"
#单张图片识别
@app.route(DETECTION_URL+'/pic',methods=['POST'])
def pic():
    if request.method == 'POST':
        now = time.time()
        data = request.get_data()
        json_data = json.loads(data)
        service = json_data['service']
        # print(service)
        if service == 'picbase':
            base = json_data['image']
            image_str = BytesIO(base64.b64decode(base))
            imgData = Image.open(image_str)
            imgData = cv2.cvtColor(numpy.asarray(imgData), cv2.COLOR_RGB2BGR)
        elif service == 'picurl':
            url = json_data['image']
            # print(url)
            response = requests.get(url)
            # print(response)
            imgData = Image.open(BytesIO(response.content))
            imgData = cv2.cvtColor(numpy.asarray(imgData), cv2.COLOR_RGB2BGR)
        else:
            return {"success": False}
        result, name = a.detect([imgData])  # 算法检测接口
        # print('花费了'+str(time.time()-now)+'秒的时间')
        # cv2.imshow("OpenCV", result[0][0])
        # cv2.waitKey()
        listtext1 = []
        if result:
            # j = json.dumps(obj=result)
            # {"success": True, "data": listtext1}
            image = Image.fromarray(cv2.cvtColor(result[0][0], cv2.COLOR_BGR2RGB))
            for i in result[0][1]:
                if i[0] == 4:
                    carpic = image.crop((i[1][0], i[1][1], i[1][2], i[1][3]))
                    # carpic.show()
                    dictsample = {"class": str(name[i[0]]), "score": i[2],
                                  "Boxes": i[1],
                                  "carnumber": bendishibie(carpic)}
                elif i[0] == 5:
                    carpic = image.crop((i[1][0], i[1][1], i[1][2], i[1][3]))
                    # carpic.show()
                    dictsample = {"class": str(name[i[0]]), "score": i[2],
                                  "Boxes": i[1],
                                  "carnumber": bendishibie(carpic)}
                else:
                    carpic = image.crop((i[1][0], i[1][1], i[1][2], i[1][3]))
                    # carpic.show()
                    dictsample = {"class": str(name[i[0]]), "score": i[2],
                                  "Boxes": i[1]}
                listtext1.append(dictsample)
            sample = [{"class": "puromisu", "score": "0.8895045", "h": 86, "w": 50, "x": 50, "y": 50},
                      {"class": "puromisu2", "score": "0.85214925", "h": 86, "w": 50, "x": 50, "y": 50}]
            # print(j)
            return {"success": True,"data": listtext1}
        else:
            return 'null'
    else:
        return 'null'
#视频录像识别接口
@app.route(DETECTION_URL+'/videotape',methods=['POST'])
def videotape2():
    data = request.get_data()
    json_data = json.loads(data)
    url = json_data['video']
    encoded = json_data['encoded']
    start_time = 0
    carnum = ""
    try:
        carnum = json_data['carnum']
    except:
        pass
    try:
        start_time = json_data['start_time']
    except:
        pass
    c = 1
    # executor2.submit(run_job2, data=[carnum,encoded,url,start_time])
    executor2.submit(run_job2, carnum=carnum, encoded=encoded, url=url, start_time=start_time)
    print(url, encoded, start_time, carnum)
    return 'ok'
if __name__ == '__main__':
    parser1 = argparse.ArgumentParser(description='manual to this script')
    parser1.add_argument('--port', type=int, default=12907)
    args1 = parser1.parse_args()
    app.run(host='0.0.0.0', port=args1.port,debug=False)