#!/usr/bin/env python
import rospy
import numpy as np
import cv2
from std_msgs.msg import UInt8MultiArray

# Define image dimensions (modify based on your actual image size)
IMG_HEIGHT = 480  # Adjust as per your data
IMG_WIDTH = 640   # Adjust as per your data

def callback(msg):
    try:
        # Convert UInt8MultiArray byte data to NumPy array
        raw_array = np.frombuffer(msg.data, dtype=np.uint8)
        if raw_array.size != IMG_HEIGHT * IMG_WIDTH:
            rospy.logerr(f"Received data size {raw_array.size} does not match expected size {IMG_HEIGHT * IMG_WIDTH}")
            return

        # Convert to float32 and adjust values
        image_array = raw_array.copy().astype(np.float32)
        image_array -= 128  # Center around 0
        image_array *= 0.2  # Scale factor (adjust as needed)

        # Reshape to image dimensions
        image_array = image_array.reshape((IMG_HEIGHT, IMG_WIDTH))

        # Separate positive and negative components
        pos_img = np.clip(image_array, 0, None)    # positive part
        neg_img = np.clip(-image_array, 0, None)   # negative part

        # Normalize each channel to 0-255 (if needed)
        if pos_img.max() > 0:
            pos_img_norm = (pos_img / pos_img.max() * 255).astype(np.uint8)
        else:
            pos_img_norm = np.zeros_like(pos_img, dtype=np.uint8)

        if neg_img.max() > 0:
            neg_img_norm = (neg_img / neg_img.max() * 255).astype(np.uint8)
        else:
            neg_img_norm = np.zeros_like(neg_img, dtype=np.uint8)

        # Create a composite color image:
        #   Blue channel  (index 0) = neg_img_norm  (negative events)
        #   Green channel (index 1) = zeros
        #   Red channel   (index 2) = pos_img_norm  (positive events)
        composite_img = np.stack([
            neg_img_norm,
            np.zeros_like(pos_img_norm),
            pos_img_norm
        ], axis=-1)

        # Display the composite image using OpenCV
        cv2.imshow("Event Image (Red=Positive, Blue=Negative)", composite_img)
        cv2.waitKey(1)  # Required to update the OpenCV window

    except Exception as e:
        rospy.logerr(f"Error processing image: {e}")

def listener():
    rospy.init_node('event_visualizer', anonymous=True)
    rospy.Subscriber("/output/event", UInt8MultiArray, callback)
    rospy.spin()  # Keep the node running

if __name__ == '__main__':
    listener()
    cv2.destroyAllWindows()


# #!/usr/bin/env python
# import rospy
# import numpy as np
# import cv2
# from std_msgs.msg import UInt8MultiArray

# # Define image dimensions (adjust these to match your data)
# IMG_HEIGHT = 480
# IMG_WIDTH = 640

# def callback(msg):
#     try:
#         # Convert the UInt8MultiArray byte data into a NumPy array
#         raw_array = np.frombuffer(msg.data, dtype=np.uint8)
#         if raw_array.size != IMG_HEIGHT * IMG_WIDTH:
#             rospy.logerr("Received data size %d does not match expected size %d",
#                          raw_array.size, IMG_HEIGHT * IMG_WIDTH)
#             return

#         # Convert to float32 so that arithmetic works properly
#         image_array = raw_array.astype(np.float32)

#         # Center the data around 0 and scale it.
#         # (These numbers may be adjusted based on your sensor/processing specifics)
#         image_array -= 128  
#         image_array *= 0.2  

#         # Reshape to the expected image dimensions
#         image_array = image_array.reshape((IMG_HEIGHT, IMG_WIDTH))

#         # For single-channel visualization we use the absolute value (magnitude) of the events.
#         abs_events = np.abs(image_array)

#         # Normalize to the range 0-255.
#         if abs_events.max() > 0:
#             norm_img = (abs_events / abs_events.max() * 255).astype(np.uint8)
#         else:
#             norm_img = np.zeros_like(abs_events, dtype=np.uint8)

#         # Display the normalized grayscale image
#         cv2.imshow("Event Image (Grayscale)", norm_img)
#         cv2.waitKey(1)  # Needed for OpenCV window to refresh

#     except Exception as e:
#         rospy.logerr("Error processing image: %s", str(e))

# def listener():
#     rospy.init_node('event_visualizer', anonymous=True)
#     rospy.Subscriber("/output/event", UInt8MultiArray, callback)
#     rospy.spin()
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     listener()
