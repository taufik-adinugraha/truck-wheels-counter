from libs import config
from libs import detection
import numpy as np
import argparse
import imutils
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="", help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="", help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1, help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

weightsPath_yolococo = 'models/yolov3.weights'
configPath_yolococo = 'models/yolov3.cfg'

weightsPath = 'models/net_kendaraan.weights'
configPath = 'models/net_kendaraan.cfg'

net_yolococo = cv2.dnn.readNetFromDarknet(configPath_yolococo, weightsPath_yolococo)
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# get the last layer
ln_yolococo = net_yolococo.getLayerNames()
ln_yolococo = [ln_yolococo[i[0] - 1] for i in net_yolococo.getUnconnectedOutLayers()]
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# check if we are going to use GPU
if config.USE_GPU:
	# set CUDA as the preferable backend and target
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None
fps = int(vs.get(cv2.CAP_PROP_FPS))

# loop over the frames from the video stream
iframe = 0
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end of the stream
	if not grabbed: 
		break

	# resize the frame and then detect object in it
	# frame = imutils.resize(frame, width=800)
	results = detection.detect_object(frame, net_yolococo, ln_yolococo, [2,5,7])

	if len(results) != 0:
		# loop over the results
		for result in results:
		    _, _, bbox, _ = result
		    
		    color = (150, 255, 255)
		    font = .7
		    (startX, startY, endX, endY) = bbox

		    vehicle = frame[startY:endY,startX:endX]
		    if vehicle.shape[0] * vehicle.shape[1] == 0:
		    	continue

		    objects = detection.detect_object(vehicle, net, ln, [0])
		        
		    if len(objects)==0:
		        continue
		    
		    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		    text = f'{len(objects)} Gandar'
		    # y = startY - 10 if startY - 10 > 10 else startY + 10
		    # font_scale = 1
		    # cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
		    
		    for veh_obj in objects:
		        _, _, bbox, _ = veh_obj
		        classID, _, bbox_obj, _ = veh_obj
		        (sX, sY, eX, eY) = bbox_obj
		        (sX, sY, eX, eY) = (startX+sX, startY+sY, startX+eX, startY+eY)
		        cv2.rectangle(frame, (sX, sY), (eX, eY), (0,0,255), 2)

		    # put the label
		    font_scale = 1.2
		    font = cv2.FONT_ITALIC
		    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
		    # set the text start position
		    text_offset_x, text_offset_y = startX + 5, startY + 35
		    # make the coords of the box with a small padding of two pixels
		    box_coords = ((text_offset_x - 5, text_offset_y + 5), (text_offset_x + text_width + 5, text_offset_y - text_height - 5))
		    overlay = frame.copy()
		    cv2.rectangle(overlay, box_coords[0], box_coords[1], color, -1)
		    cv2.putText(overlay, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0,0,0), thickness=2)
		    # apply the overlay
		    alpha=0.6
		    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


	# check to see if the output frame should be displayed to our
	# screen
	if args["display"] > 0:
		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# if an output video file path has been supplied and the video
	# writer has not been initialized, do so now
	if args["output"] != "" and writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 25,
			(frame.shape[1], frame.shape[0]), True)

	# if the video writer is not None, write the frame to the output
	# video file
	if writer is not None:
		writer.write(frame)

	iframe += 1