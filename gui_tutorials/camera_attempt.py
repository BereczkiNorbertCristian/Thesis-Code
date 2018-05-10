import cv2
import numpy as np
import sys
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSignal

class ShowVideo(QtCore.QObject):
	#initiating the built in camera
	camera_port = 0
	camera = cv2.VideoCapture(camera_port)
	VideoSignal = QtCore.pyqtSignal(QtGui.QImage)


	def __init__(self, parent = None):
    	super(ShowVideo, self).__init__(parent)

	@QtCore.pyqtSlot()
	def startVideo(self):

    	run_video = True
    	while run_video:
        	ret, image = self.camera.read()

        	color_swapped_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        	height, width, _ = color_swapped_image.shape

        	#width = camera.set(CAP_PROP_FRAME_WIDTH, 1600)
        	#height = camera.set(CAP_PROP_FRAME_HEIGHT, 1080)
        	#camera.set(CAP_PROP_FPS, 15)

        	qt_image = QtGui.QImage(color_swapped_image.data,
									width,
									height,
									color_swapped_image.strides[0],
									QtGui.QImage.Format_RGB888)

			self.VideoSignal.emit(qt_image)



class ImageViewer(QtWidgets.QWidget):
	def __init__(self, parent = None):
		super(ImageViewer, self).__init__(parent)
		self.image = QtGui.QImage()
		self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)
	def paintEvent(self, event):
		painter = QtGui.QPainter(self)
		painter.drawImage(0,0, self.image)
		self.image = QtGui.QImage()

	def initUI(self):
		self.setWindowTitle('Test')

	@QtCore.pyqtSlot(QtGui.QImage)
	def setImage(self, image):
		if image.isNull():
			print("Viewer Dropped frame!")

		self.image = image
		if image.size() != self.size():
			self.setFixedSize(image.size())
		self.update()

