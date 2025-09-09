# HSV Change and Production Analysis Tool

A tool for analyzing HSV changes in images and predicting production based on production data.

## Environment Requirements

It is recommended to use **Mu Python Editor** as the development and running environment, which is designed for beginners with a simple and easy-to-use interface:
- Mu Python Editor download address: https://codewith.mu/

## Install Dependencies

Before using, please install the following necessary Python packages. Open the "Terminal" function in Mu Editor and execute the following commands:

```bash
pip install tkinter matplotlib numpy pillow scikit-learn pandas openpyxl
```

## Function Introduction

The tool mainly includes three core functional modules:
1. HSV Change Analysis - Extract HSV features from time-series images and establish a time prediction model
2. Production Model Analysis - Establish a relationship model between running time and production/indicators from Excel data
3. Comprehensive Inference Analysis - Use the above two models to infer time through new images and predict production

## Usage Steps

### 1. Prepare Data

- Prepare an Excel file containing production data, which must include the following columns:
  - Running time
  - Production
  - Indicator data

- Prepare two image folders:
  - `images_learning`: Time-series images for training, with filenames in the format `YYYY-MM-DD-HH-MM-SS.jpg`
  - `images_analysis`: Sample images for inference

### 2. Load Data and Establish Models

#### Step 1: Establish HSV-Time Model
1. Select the image directory as `images_learning`
2. Click the "Load Images" button, and the program will automatically parse the image time and calculate HSV values
3. Select the polynomial regression degree (2-5)
4. Click the "Train Model" button to generate a model of HSV changes over time

#### Step 2: Establish Production Model
1. Click the "Load Excel File" button and select the prepared production data Excel file
2. Select the polynomial order (1-5)
3. Click the "Train Model" button to generate a relationship model between running time and production/indicators

### 3. Image Inference and Production Prediction

1. In the comprehensive inference analysis module, click the "Browse" button to select a sample image from the `images_analysis` folder
2. Click the "Set ROI Area" button, and click two points on the image to select the region of interest
3. Click the "Image Inference" button, and the program will infer the corresponding running time based on HSV values
4. Click the "Predict Production Indicators" button to calculate the corresponding production and indicator data based on the inferred time

## Notes

- Image filenames must strictly follow the `YYYY-MM-DD-HH-MM-SS.jpg` format; otherwise, the time cannot be parsed correctly
- The Excel file must contain the three columns: "Running time", "Production", and "Indicator data"
- When selecting the ROI area, it is recommended to choose an area with obvious features and representative in the image
- The selection of polynomial order will affect the model accuracy; you can try different orders according to the actual data

## Output Description

- After the model training is completed, the R² score will be displayed (the closer to 1, the better the model fit)
- The inference result will display the predicted running time (in hours:minutes:seconds format)
- The production prediction will be calculated based on the inferred time and the production model
