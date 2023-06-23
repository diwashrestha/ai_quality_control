# AI_quality_control
This project implements a Steel Defect Detection System using a UNet model for image segmentation. The system is designed to detect  defects in steel images, aiding in quality control and inspection processes. It utilizes the PyTorch framework for deep learning and Streamlit for the user interface.


## Prerequisites

Before running the code, ensure that you have the following dependencies installed:

- Python 3.x
- Streamlit
- OpenCV
- PyTorch
- Pillow (PIL)
- torchvision
- matplotlib

You can install the required dependencies by running the following command:

```python
pip install -r requirements.txt
```

Run the application:

```python
streamlit run app.py
```


## Usage

1. Upload an image file (.jpg, .png, .jpeg) containing steel defects through the user interface.

2. The system will perform defect segmentation and display the results. The image will be shown along with the detected defects classified into different classes.

3. Alternatively, you can try sample images provided in the project. Click the "Try" button to randomly select a sample image and view the segmentation results.

## Model Architecture

The project includes a custom implementation of the UNet model for image segmentation. The architecture consists of encoder and decoder pathways with skip connections to preserve spatial information.


## Data Preprocessing

Data preprocessing steps such as resizing, normalization, and augmentation can be added to the project as per specific requirements. The current implementation assumes input images are preprocessed and ready for inference.

Note: Make sure to provide the correct path to the pre-trained model file in the `defect_segmentation.py` file.