### Colorization using Deep Learning

This Python script utilizes a pre-trained deep learning model to perform colorization on grayscale images. The model is based on Convolutional Neural Networks (CNNs) and utilizes OpenCV for image processing.

### Requirements
- Python 3.x
- OpenCV (`cv2`)
- NumPy

### Installation
1. Clone or download the repository.

2. Install the required Python packages using pip:

```
pip install opencv-python numpy
```

### Usage
1. Ensure you have the required model files in the specified directory:
   - `colorization_deploy_v2.prototxt`
   - `pts_in_hull.npy`
   - `colorization_release_v2.caffemodel`

2. Run the script from the command line, providing the path to the grayscale image you want to colorize using the `-i` or `--image` argument.

Example:
```
python colorize_image.py -i path/to/your/image.jpg
```

### Script Explanation
- The script loads a pre-trained deep learning model for colorization along with necessary configuration files.
- It reads the input image specified by the user.
- The image is preprocessed by converting it to the LAB color space.
- The L channel (representing the grayscale image) is extracted and resized for input to the model.
- The model predicts the 'a' and 'b' channels for the image.
- The colorized image is reconstructed in the LAB color space.
- Finally, the colorized image is displayed alongside the original image.

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
