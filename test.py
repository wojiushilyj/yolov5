
import minio
import uuid
from datetime import datetime

# http://10.18.211.186:9000/
minio_conf = {
    'endpoint': '10.18.211.186:9000',
    'access_key': 'reload',
    'secret_key': 'Abc12345678',
    'secure': False
}
def up_data_minio(bucket: str,filepath,objectname):
    client = minio.Minio(**minio_conf)
    client.fput_object(bucket_name=bucket, object_name=objectname,
                       file_path=filepath,
                       content_type='application/jpg'
                       )
# uuid_str = uuid.uuid4().hex
# save_screenshot = datetime.now().now().strftime('%Y%m%d%H%M%S') + '_%s.jpg' % uuid_str
# up_data_minio('screenshot',
#                r"D:\python\yolov5\image\0.jpg",
#                save_screenshot)
shotsavetime = '0'
gaptime=int(datetime.now().now().strftime('%Y%m%d%H%M%S')) - int(shotsavetime)
print(gaptime)
