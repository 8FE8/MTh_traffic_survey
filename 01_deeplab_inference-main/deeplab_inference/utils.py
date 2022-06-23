import tensorflow as tf

def resize_label(label, label_size):
    """Resizes a label to label_size.
    
    Takes a label and resizes it to label_size by using the nearest neighbor
    method.
    
    Args:
        label: A label of raw segmentations of size [height, width].
        label_size: The new size of the label is [height, width].
        
    Returns:
        A label of raw segmentations of size label_size.
    """
    label = tf.expand_dims(label, axis=2)
    resized_label = tf.image.resize(label, label_size, method='nearest')
    
    return tf.squeeze(resized_label, axis=2)