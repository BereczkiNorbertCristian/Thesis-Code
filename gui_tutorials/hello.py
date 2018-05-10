import sys
import cv2
import time
import numpy as np

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget, QImage, QPixmap
from PyQt5.QtCore import QSize    
from PyQt5.QtCore import QThread, QObject, pyqtSignal, pyqtSlot

class ImageStreamer(QObject):
	finished = pyqtSignal()
	frame_ready = pyqtSignal(np.ndarray)

	@pyqtSlot()
	def stream_frames(self):
		cap = cv2.VideoCapture(0)
		while(True):
			ret, frame = cap.read()
			frame = frame[:,::-1]
			self.frame_ready.emit(frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		cap.release()
		cv2.destroyAllWindows()
		self.finished.emit()


class HelloWindow(QMainWindow):
	def __init__(self):
		QMainWindow.__init__(self)

		self.obj = ImageStreamer()
		self.thread = QThread()

		self.obj.frame_ready.connect(self.on_frame_ready)
		self.obj.moveToThread(self.thread)
		self.obj.finished.connect(self.thread.quit)
		self.thread.started.connect(self.obj.stream_frames)

		self.thread.start()

		self.initUI()

	def initUI(self):
		self.setMinimumSize(QSize(640, 480))    
		self.setWindowTitle("Hello world") 

		centralWidget = QWidget(self)          
		self.setCentralWidget(centralWidget)   

		gridLayout = QGridLayout(self)     
		centralWidget.setLayout(gridLayout)  
 
		title = QLabel("Hello World from PyQt", self) 
		title.setAlignment(QtCore.Qt.AlignCenter) 
		gridLayout.addWidget(title, 0, 0)

		self.image_label = QLabel()

	def on_frame_ready(self,frame):
		image = QImage(
			frame,
			frame.shape[1],
			frame.shape[0],
			frame.shape[1] * 3,
			QImage.Format_RGB888)
		self.image_label.setPixmap(QPixmap.fromImage(image))

 
if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	mainWin = HelloWindow()
	mainWin.show()
	sys.exit( app.exec_() )