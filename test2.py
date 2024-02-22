# import torch
# import numpy as np
# from ultralytics import YOLO

# # Load a pre-trained instance segmentation model
# model = YOLO('sidewalk_segment.pt')  

# # Inference on an image
# results = model('sidewalk.png', stream = True)

# for result in results:
#     # Display segmentation results
#     result.show()   

#     # print(result.masks.xy[0])
    
    
#     mask_coordinate = result.masks.xy[0].cpu().numpy()  # DataFrame with bounding boxes, classes, confidences, masks
#     print(mask_coordinate)



import torch
import numpy as np
from ultralytics import YOLO

model = YOLO('sidewalk_segment.pt')

# Inference on an image
results = model('sidewalk.png', stream = True)

for result in results:
    result.show()

    # Get the first mask's coordinates (as a PyTorch tensor)
    mask_coords = result.masks.xy[0]  

    # Convert mask coordinates to CPU (if needed)
    if mask_coords.cuda:  # Check if it's on the GPU
        mask_coords = mask_coords.cpu()

    # Convert to NumPy array 
    mask_coords_numpy = mask_coords.numpy() 

    # Calculate the area of the mask
    area = np.poly_area(mask_coords_numpy[:, 0], mask_coords_numpy[:, 1])
    print(f"Area of the mask: {area}") 


