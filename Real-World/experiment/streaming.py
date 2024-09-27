import cv2
from PyQt5.QtWidgets import QMainWindow, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
import numpy as np
from camera import process_image, get_seeker_pixel  # Assuming these are your custom functions

image_size = 960

def get_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    frame = process_image(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

class Worker(QObject):
    update = pyqtSignal(np.ndarray)  # Signal to emit the processed frame
    finished = pyqtSignal()  # Signal to indicate that processing is done

    def __init__(self, cap, acccam_list, vicon_offset, seeker_location, agent_queue,frame_list,seeker_action):
        super().__init__()
        self.cap = cap
        self.acccam_list = acccam_list
        self.vicon_offset = vicon_offset
        self.seeker_location = seeker_location
        self.agent_queue = agent_queue
        self.frame_list = frame_list
        self.running = True
        self.seeker_action = seeker_action
        self.image_size = image_size

    def run(self):
        while self.running:
            frame = None
            frame2 = get_frame(self.cap)
            if frame2 is not None:
                
                for i, seeker_id in enumerate(self.agent_queue):
                    acccam = self.acccam_list[seeker_id-1]
                    frame1, seeker_pixel = get_seeker_pixel(frame2, self.seeker_location[seeker_id-1], self.vicon_offset)

                    if i == 0:
                        frame4 = acccam.add_mask(frame1, seeker_pixel)
                        frame = frame4.clone()
                        
                        position = self.seeker_location[seeker_id-1]
                        action = self.seeker_action[seeker_id-1]
                        
                        if position is not None:
                            self_y = int((self.vicon_offset[0] - position[0])/(2*self.vicon_offset[0]) * self.image_size)
                            self_x = int((self.vicon_offset[1] + position[1])/(2*self.vicon_offset[1]) * self.image_size)
                            for a in range(self_x-13,self_x+14):
                                for b in range(self_y-13,self_y+14):
                                    if a >= 0 and a < self.image_size and b >= 0 and b < self.image_size:
                                        if frame.shape[0] == 3:
                                            frame[0,a,b] = 0.5
                                            frame[1,a,b] = 0
                                            frame[2,a,b] = 0.5
                                        else:
                                            frame[:,0,a,b] = 0.5
                                            frame[:,1,a,b] = 0
                                            frame[:,2,a,b] = 0.5
      
                        for j in range(1, len(self.agent_queue)):
                            teammate_id = self.agent_queue[j]
    
                            teammate_position = self.seeker_location[teammate_id-1]
                            
                            if teammate_position is not None:
                                if abs(teammate_position[0] - position[0]) >= 1.5 or abs(teammate_position[1] - position[1]) >= 1.5:
                                    y = int((self.vicon_offset[0] - teammate_position[0])/(2*self.vicon_offset[0]) * self.image_size)
                                    x = int((self.vicon_offset[1] + teammate_position[1])/(2*self.vicon_offset[1]) * self.image_size)

                                    for j in range(x-13,x+14):
                                        for m in range(y-13,y+14):
                                            if j >= 0 and j < self.image_size and m >= 0 and m < self.image_size:
                                                if frame.shape[0] == 3:
                                                    frame[0,j,m] = 1
                                                    frame[1,j,m] = 0
                                                    frame[2,j,m] = 0
                                                else:
                                                    frame[:,0,j,m] = 1
                                                    frame[:,1,j,m] = 0
                                                    frame[:,2,j,m] = 0
                                
                    
                        
                        self.frame_list[seeker_id-1] = frame4
                    else:
                        frame3 = acccam.add_mask(frame1, seeker_pixel)
                        self.frame_list[seeker_id-1] = frame3

                frame = frame.squeeze().numpy().transpose(1, 2, 0)
                
                self.update.emit(frame)

        self.finished.emit()

    def stop(self):
        self.running = False

class VideoStreamUI(QMainWindow):
    def __init__(self, cap, num_of_seekers, acccam_list):
        super().__init__()
        self.initUI()

        self.cap = cap
        self.vicon_offset = [2.66, 2.5]
        self.agent_queue = [i+1 for i in range(num_of_seekers)]
        self.acccam_list = acccam_list
        self.seeker_location = [None for _ in range(num_of_seekers)]
        
        self.human_control = [False for _ in range(num_of_seekers)]
        self.human_action = [None for _ in range(num_of_seekers)]
        self.frame_list = [None for _ in range(num_of_seekers)]
        self.action_list = [None for _ in range(num_of_seekers)]

        self.thread = QThread()
        self.worker = Worker(self.cap, self.acccam_list, self.vicon_offset, self.seeker_location, self.agent_queue, self.frame_list, self.action_list)
        self.worker.moveToThread(self.thread)

        
        # Connect signals and slots
        self.thread.started.connect(self.worker.run)
        self.worker.update.connect(self.update_frame)
        self.worker.finished.connect(self.thread.quit)
        self.thread.start()
    
    def update_seeker_location(self, seeker_info_dict, seeker_action_dict):
        for seeker_id, seeker_info in seeker_info_dict.items():
            seeker_id = int(seeker_id[-1])
            if seeker_id in self.agent_queue and seeker_info is not None and seeker_info != {}:
                self.seeker_location[seeker_id-1] = seeker_info["location"]

        
        for seeker_id, seeker_action in seeker_action_dict.items():
            if seeker_id.find("Seeker") != -1:
                seeker_id = int(seeker_id[-1])
                if seeker_id <= len(self.action_list):
                    self.action_list[seeker_id-1] = seeker_action
        
        self.update_human_control_info()
    
    def update_human_control_info(self):
        for i in range(len(self.human_control)):
            if self.human_control[i] and ((self.human_action[i][0] - self.seeker_location[i][0])**2+(self.human_action[i][1] - self.seeker_location[i][1])**2)**0.5 < 0.3:
                self.human_control[i] = False
                self.human_action[i] = None
                

    def initUI(self):
        self.setWindowTitle('Camera Stream with Click Detection')
        self.label = QLabel(self)
        self.setCentralWidget(self.label)
        self.setGeometry(500, 720, image_size, image_size)

    def update_frame(self, frame):
        frame = (frame * 255).astype(np.uint8).tobytes()
        height, width = image_size, image_size
        bytes_per_line = 3 * width
        qimg = QImage(frame, width, height, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qimg))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            x = event.pos().x()
            y = event.pos().y()
        
            if x <= 482:
                map_x = (482 - x)/(482-19)* 2.5
            else:
                map_x = -(x - 482)/(943-482) * 2.5

            if y >= 477:
                map_y = (y-477)/(943-477) * 2.5
            else:
                map_y = -(477-y)/(477-20) * 2.5
                

            seeker_id = self.agent_queue[0] - 1
            self.human_action[seeker_id] = (map_x, map_y)
            self.human_control[seeker_id] = True
        elif event.button() == Qt.RightButton:
            self.agent_queue.append(self.agent_queue.pop(0))

    def closeEvent(self, event):
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()
        self.cap.release()
        event.accept()

