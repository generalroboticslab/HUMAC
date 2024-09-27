import time
from robomaster import conn
from MyQR import myqr
from PIL import Image

QRCODE_NAME = "qrcode.png"

if __name__ == '__main__':

    helper = conn.ConnectionHelper()
    info = helper.build_qrcode_string(ssid="VICON_Server_5G", password="duke-robotics1")
    myqr.run(words=info)
    time.sleep(1)
    img = Image.open(QRCODE_NAME)
    img.show()
    if helper.wait_for_connection():
        print("Connected!")
        img.close()
    else:
        print("Connect failed!")
        img.close()
    
