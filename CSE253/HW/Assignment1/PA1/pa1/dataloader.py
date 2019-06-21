################################################################################
# CSE 253: Programming Assignment 1
# Code snippet by Jenny Hamer
# Winter 2019
################################################################################
# We've provided you with the dataset in CAFE.tar.gz. To uncompress, use:
# tar -xzvf CAFE.tar.gz
################################################################################
# To install PIL, refer to the instructions for your system:
# https://pillow.readthedocs.io/en/5.2.x/installation.html
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################

from os import listdir
from PIL import Image
import numpy as np


# The relative path to your CAFE-Gamma dataset
data_dir = "./CAFE/"

# Dictionary of semantic "label" to emotions
emotion_dict = {"h": "happy", "ht": "happy with teeth", "m": "maudlin",
	"s": "surprise", "f": "fear", "a": "anger", "d": "disgust", "n": "neutral"}


def convert_to_one_hot(Y, C):
    """
    convert labels to one-hot
    """
    
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def load_data(mode):
    """
    mode: "happy_maudlin" or "anger_surprised" or "all"
    return: dataset
    """
    
    data_dir = "./CAFE/"
    all_files = []
    images = []
    labels = []
    pid = []
    pid_dict = {'018':0,'027':1,'036':2,'037':3,'041':4,
                '043':5,'044':6,'048':7,'049':8,'050':9}
    emotion_label = None
    if mode == "happy_maudlin":
        all_dirs = ['h', 'm']
        label_dict = {'h':1, 'm':0}
    elif mode == "anger_surprised":
        all_dirs = ['a', 's']
        label_dict = {'a':1, 's':0}
    else:
        label_dict = {'a':0,'d':1,'f':2,'h':3,'m':4,'s':5}
        all_dirs = ['a', 'd', 'f', 'h', 'm', 's']
            
    for dirs in all_dirs:
        files = listdir(data_dir + dirs)
        while '.DS_Store' in files:
            files.remove('.DS_Store')
        
        all_files += [dirs+'/'+name for name in files]
        labels += [label_dict[dirs]] * len(files)
    
    # Store the images as arrays and their labels in two lists
    
    for file in all_files:
        # Load in the files as PIL images and convert to NumPy arrays
        img = Image.open(data_dir + file)
        images.append(np.array(img))
        pid.append(file.split('_')[0].split('/')[1])
    
    pids = [pid_dict[x] for x in pid]
    pids = np.array(pids)

    emotion_label = labels
    labels = np.expand_dims(np.array(labels), axis=1)
    if mode != "happy_maudlin" and mode != "anger_surprised":
        labels = convert_to_one_hot(labels, 6).T
    
    images = np.array(images).astype('float64')
    # vectorization
    images = np.reshape(np.array(images), (len(images), -1))
    
    return images, labels, pids, np.array(emotion_label)
    

def display_face(img):
    """ Display the input image and optionally save as a PNG.

    Args:
        img: The NumPy array or image to display

    Returns: None
    """
    # Convert img to PIL Image object (if it's an ndarray)
    if type(img) == np.ndarray:
        print("Converting from array to PIL Image")
        img = Image.fromarray(img)

	# Display the image
    img.show()



def load_data_original(data_dir="./CAFEORI/"):
	""" Load all PGM images stored in your data directory into a list of NumPy
	arrays with a list of corresponding labels.

	Args:
		data_dir: The relative filepath to the CAFE dataset.
	Returns:
		images: A list containing every image in CAFE as an array.
		labels: A list of the corresponding labels (filenames) for each image.
	"""
	# Get the list of image file names
	all_files = listdir(data_dir)

	# Store the images as arrays and their labels in two lists
	images = []
	labels = []

	for file in all_files:
		# Load in the files as PIL images and convert to NumPy arrays
		img = Image.open(data_dir + file)
		images.append(np.array(img))
		labels.append(file)

	print("Total number of images:", len(images), "and labels:", len(labels))

	return images, labels