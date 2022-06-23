# General setting
# video path
video_path = 'exclude/DJI_0008.MP4'
# Output frame dimensions
frame_width = 1920
frame_height = 1080

# Background subtraction
threshold = 180   # select a value between 0 and 255
kernel_dilatation = (5, 5)  # Select dimension of first kernel dilatation

# Optical Flow
# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners=300, qualityLevel=0.2, minDistance=7, blockSize=7)

# Param to filter the bounding boxes
min_width = 15  # in pixels
max_width = 400  # in pixels
min_height = 15  # in pixels
max_height = 400  # in pixels
max_disappeared = 5

# Param to filter the path
min_life = 4  # in frames
min_lenght = 10  # in nr of element
min_distance = 15  # in meters
