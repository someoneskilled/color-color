import numpy as np
import argparse
import cv2
import os

# Define paths to model files
PROTOTXT = "models/colorization_deploy_v2.prototxt"
POINTS = "models/pts_in_hull.npy"
MODEL = "models/colorization_release_v2.caffemodel"

# Argparse to accept input image path
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
    help="Path to input image")
args = vars(ap.parse_args())

# Load the pre-trained model
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)

# Prepare model for colorization
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Load the input image
image = cv2.imread(args["image"])

# Preprocess the image
scaled = image.astype("float32") / 255.0
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
resized = cv2.resize(lab, (224, 224))
L = cv2.split(resized)[0]
L -= 50

# Colorize the image
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

# Combine L channel and colorized ab channels
L = cv2.split(lab)[0]
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

# Convert LAB to BGR
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)
colorized = (255 * colorized).astype("uint8")

# Display original and colorized images
cv2.imshow("Original", image)
cv2.imshow("Colorized", colorized)
cv2.waitKey(0)
