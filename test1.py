# import cv2
# from ultralytics import YOLO

# # Load the YOLOv8 model
# model = YOLO('sidewalk_segment.pt')

# results = model(source = 'sidwalk.png', show = False, stream = True, device = 0)

# for result in results:
#     result.show()
#     print(result.masks)



# import cv2
# import numpy as np
# from ultralytics import YOLO

# # Load the YOLOv8 model for sidewalk segmentation
# model = YOLO('sidewalk_segment.pt')

# # Run inference on an image
# results = model(source='sidwalk.png', show=False, stream=True, device=0)

# # Assuming results contain one primary mask for the sidewalk
# # Convert the mask tensor to a numpy array and then to a binary image
# # Note: Adjust the thresholding as per your mask's pixel value range for the sidewalk
# for result in results:
#     mask = result.masks.data.cpu().numpy()[0]  # Assuming there's only one mask and converting it to numpy
#     binary_mask = (mask > 0).astype(np.uint8)  # Convert mask to binary: sidewalk=1, otherwise=0

#     # Find the contours in the binary image
#     contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Assuming the largest contour corresponds to the sidewalk
#     # This part may need adjustment based on your specific scenario
#     largest_contour = max(contours, key=cv2.contourArea)
    
#     # Calculate the centroid of the largest contour
#     M = cv2.moments(largest_contour)
#     if M["m00"] != 0:
#         cx = int(M["m10"] / M["m00"])
#         cy = int(M["m01"] / M["m00"])
#         center_point = (cx, cy)
#         print(f"Centroid of the sidewalk: {center_point}")
#     else:
#         print("No sidewalk detected.")

#     # Optional: Visualize the centroid on the original image
#     # Load the original image
#     image = cv2.imread('sidewalk.png')
#     # Draw the centroid
#     cv2.circle(image, center_point, radius=5, color=(0, 255, 0), thickness=-1)
#     cv2.imshow("Centroid", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()



import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model for sidewalk segmentation
model = YOLO('sidewalk_segment.pt')

# Run inference on an image
results = model(source='sidwalk.png', show=False, stream=True, device=0)

# Iterate through each detected object in the results
for result in results:
    # Make sure to check if masks exist for the detected objects
    if hasattr(result, 'masks') and result.masks.data.numel() > 0:
        mask = result.masks.data.cpu().numpy()[0]  # Assuming there's only one mask and converting it to numpy
        binary_mask = (mask > 0).astype(np.uint8)  # Convert mask to binary: sidewalk=1, otherwise=0

        # Find the contours in the binary image
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Assuming the largest contour corresponds to the sidewalk
            largest_contour = max(contours, key=cv2.contourArea)

            # Calculate the centroid of the largest contour
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                center_point = (cx, cy)
                print(f"Centroid of the sidewalk: {center_point}")

                # Optional: Visualize the centroid on the original image
                # Load the original image
                image = cv2.imread('sidwalk.png')
                # Draw the centroid
                cv2.circle(image, center_point, radius=5, color=(0, 255, 0), thickness=-1)
                cv2.imshow("Centroid", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("No sidewalk detected.")
        else:
            print("No contours found.")
    else:
        print("No masks found for this result.")
