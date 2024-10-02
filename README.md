# Computer Vision

## Project Overview

This repository contains two laboratory exercises from the **Computer Vision** course. 
The labs explore key concepts such as edge detection, interest point detection, optical flow, and image stitching using Python.


## Table of Contents
- [Lab 1: Edge and Interest Point Detection](#lab-1-edge-and-interest-point-detection)
- [Lab 2: Optical Flow and Image Stitching](#lab-2-optical-flow-and-image-stitching)
- [Features](#features)

## Lab 1: Edge and Interest Point Detection

**Objective:**  
This lab focuses on implementing various edge and interest point detection techniques, including edge detection using Laplacian of Gaussian and interest point detection using the Harris and Hessian-Laplace methods.

**Key Tasks:**
1. **Edge Detection:**
   - Implement edge detection using Gaussian and Laplacian-of-Gaussian filters.
   - Apply the algorithms to both synthetic and real images to observe the effects of noise and filter size.
   - Evaluate the quality of detected edges using quantitative measures such as precision and recall.

2. **Interest Point Detection:**
   - Implement Harris and Hessian-Laplace methods for detecting corners and blobs in images.
   - Visualize the detected points using a provided Python function.
   - Experiment with multi-scale interest point detection and compare the results.


## Lab 2: Optical Flow and Image Stitching

**Objective:**  
This lab covers optical flow estimation for tracking moving objects in video sequences and image stitching to create panoramic images.

**Key Tasks:**
1. **Optical Flow (Lucas-Kanade Method):**
   - Implement the Lucas-Kanade algorithm for tracking face and hand movements in sign language videos.
   - Compare the results with other optical flow methods such as TV-L1.
   
2. **Image Stitching:**
   - Stitch multiple images together using feature matching and homography estimation to create a seamless panoramic image.
   - Implement RANSAC to improve the robustness of the stitching process.

## Features

- **Edge Detection:** Various edge detection methods, including Laplacian of Gaussian and morphological operators.
- **Interest Point Detection:** Implementation of multi-scale Harris and Hessian detectors.
- **Optical Flow Tracking:** Use Lucas-Kanade and TV-L1 methods for object tracking in video sequences.
- **Image Stitching:** Stitching of multiple images to create panoramic views.

## License

This project is licensed under the Apache 2.0 License.
