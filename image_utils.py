import pandas as pd
import numpy as np
from PIL import Image
import torch
import defect_segmentation
from torchvision import transforms, utils, datasets
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import cv2



def compare_arrays(array1, array2, threshold):
    mae = np.mean(np.abs(array1 - array2))
    
    if mae <= threshold:
        return True
    else:
        return False


defect_classify_model = load_model('model/multilabel_model5v2.h5')

def preprocess_test_image(image_dir, image_size):
    image = cv2.imread(image_dir)
    image = cv2.resize(image, (image_size[1], image_size[0]))

    image = image /255.0
    image = np.expand_dims(image, axis=0)
    return image

def defect_check(image_path, model = defect_classify_model):
    test_image = preprocess_test_image(image_path, (256, 160))
    prediction = model.predict(test_image)
    predicted_labels = (prediction > 0.5).astype(int)
    predicted_class_indices = np.where(predicted_labels[0] == 1)[0]
    
    return(predicted_class_indices.item())





def image_inference(image_path, threshold, model_cls):

    # Dictionary mapping model class to their respective paths
    model_paths = {
        1: 'model/seg_1_model.pt',
        2: 'model/seg_2_model.pt',
        3: 'model/seg_3_model.pt',
        4: 'model/seg_4_model.pt'
    }
    
    # Check the availability of CUDA and choose device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the specified model
    model = torch.load(model_paths[model_cls], map_location=torch.device('cpu'))
    model.eval()

    # Define the cropping positions
    img_crop_mat = [0, 260,520, 780, 1040, 1300]

    # Open the image
    image = Image.open(image_path)

    # Normalize constants for the image
    MEAN =  99.45881939063021
    STD =  50.285401061795476
    m = MEAN/255
    s = STD/255

    # Transformations for the image
    t1 = transforms.ToTensor()
    t2 = transforms.Normalize(mean = m , std = s)
    
    # Apply transformations to the image
    img_tensor = t2(t1(image)[0].unsqueeze(0))
    

    for i in range(len(img_crop_mat)):
        p = img_crop_mat[i]

        # Crop the image tensor
        X_ = img_tensor[:, :, p : p + 256]

        # Set the model to evaluation mode and make predictions
        model.eval()
        with torch.no_grad():
            y_pred = model(X_.unsqueeze(0).to(device))
        pred_mask = y_pred.cpu().numpy().squeeze()

        # Extract the cropped image matrix
        img_matrix = (mpimg.imread(image_path).transpose(-1,0,1)[:,:, p : p +256].transpose(1,2,0))
        

         # Create a copy of the image matrix for predictions
        img_pred = img_matrix.copy()
        img_pred = img_pred.transpose(-1,0,1)
        
        # Apply different color segmentation for different classes
        if model_cls == 1:
            img_pred[0, pred_mask >= threshold] = 209
            img_pred[1, pred_mask >= threshold] = 7
            img_pred[2, pred_mask >= threshold] = 8
        elif model_cls == 2:
            img_pred[0, pred_mask >= threshold] = 6
            img_pred[1, pred_mask >= threshold] = 208
            img_pred[2, pred_mask >= threshold] = 8   
        elif model_cls == 3:
            img_pred[0, pred_mask >= threshold] = 6
            img_pred[1, pred_mask >= threshold] = 208
            img_pred[2, pred_mask >= threshold] = 8    
        elif model_cls == 4:
            img_pred[0, pred_mask >= threshold] = 63
            img_pred[1, pred_mask >= threshold] = 107
            img_pred[2, pred_mask >= threshold] = 208

        img_pred = img_pred.transpose(1,2,0)
    
        pred = 1 * (pred_mask >= threshold)
    
        
         # Concatenate the image matrices
        if 'final_img_matrix' not in locals():
            # Create a new variable with a value
            final_img_matrix = img_matrix.copy()
        else:
                # Concatenate another list or value
            final_img_matrix = np.concatenate((final_img_matrix, img_matrix), axis=1)
            
        if 'final_img_pred' not in locals():
            # Create a new variable with a value
            final_img_pred = img_pred.copy()
        else:
                # Concatenate another list or value
            final_img_pred = np.concatenate((final_img_pred, img_pred), axis=1)

        
    return [final_img_matrix, final_img_pred]



