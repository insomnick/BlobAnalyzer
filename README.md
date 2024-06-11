# BlobAnalyzer: Advanced Fungal Culture Analysis

## 1. Introduction
BlobAnalyzer is innovative software designed to analyze the growth of fungal cultures, specifically focusing on *Physarum polycephalum*, the yellow blob. The application automates measuring growth by analyzing images of petri dishes before and after experiments, ensuring precise and efficient data collection.

## 2. Objectives
The primary objective of BlobAnalyzer is to streamline the analysis of *Physarum polycephalum* growth in petri dishes. The software aims to:
- Automate image alignment and size comparison using ArUco markers.
- Precisely analyze the yellow growth within a defined pink circle.
- Calculate and compare the growth of the blob before and after the experiment.
- Output results into a CSV file, including measurements and inaccuracies.

## 3. Feature Pipeline
load image -> correct perspective -> pink_mask + circle detection
->  detect blob (yellow) -> ratio from AruUco markers -> data -> success

## 4. Technology Stack
- **Programming Languages:** Python
- **Libraries:** OpenCV (image processing), NumPy (numerical operations), Pandas (data manipulation)
- **Tools:** ArUco markers for image alignment

## 5. User Guide
### Getting Started
- Install BlobAnalyzer by cloning the repository from GitHub and running the setup script.
- Example command: `git clone [repository URL] && cd BlobAnalyzer && pip install -r requirements.txt`

### Using BlobAnalyzer
- Upload the images of the petri dish taken before and after the experiment.
- The software will automatically align the images using ArUco markers.
- It will then analyze the yellow growth within the pink circle, calculate the size, and compare the results.
- The final results will be written to a CSV file, including measurement inaccuracies.

## 6. Conclusion
BlobAnalyzer offers a significant advancement in the analysis of *Physarum polycephalum* growth. By automating the image alignment, growth measurement, and comparison process, the software enables researchers to obtain precise and reliable data efficiently. This tool not only saves time but also enhances the accuracy of experimental results.
