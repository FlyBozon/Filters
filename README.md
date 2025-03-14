# Cartoon Filters
## Overview  

This project explores the implementation of a **cartoon-style video filter** using **CUDA** for optimized performance. Initially, the filter was developed and tested using **Python** and **OpenCV**, which provided a baseline for achieving the desired cartoon effect. However, due to performance limitations in real-time processing, CUDA was introduced to accelerate the computation and optimize the filter's execution.  

Additionally, to complement the cartoon visual style, an **audio modification feature** was incorporated. This transforms the video's audio to match the playful, exaggerated nature of cartoon animations, enhancing the overall effect.  

## Features  

- **Cartoon Filter Implementation**: Converts videos into a stylized cartoon-like format by applying edge detection, color quantization, and smoothing techniques.  
- **CUDA Optimization**: Speeds up the filtering process significantly, enabling real-time or near real-time performance.  
- **Python + OpenCV Prototype**: Initial development and testing were conducted in Python to refine the filtering technique before porting it to CUDA.  
- **Audio Effects**: Enhances the video's sound to match the cartoonish visuals, making the final output more immersive and engaging.  

## How It Works  

1. **Preprocessing**: The input video is read and converted into frames.  
2. **Edge Detection**: Detects significant edges to highlight the cartoon effect.  
3. **Color Quantization**: Reduces the number of colors to create a more stylized, posterized look.  
4. **Smoothing & Stylization**: Applies filters to create a hand-drawn appearance.  
5. **CUDA Acceleration**: Optimizes these steps to achieve faster performance.  
6. **Audio Processing**: Modifies the audio to match the cartoon theme.  
7. **Final Output**: The modified frames and audio are recombined into a new cartoon-styled video.  

## Why CUDA?  

CUDA allows for parallel processing on **NVIDIA GPUs**, significantly improving the performance of computationally expensive image-processing tasks. By leveraging CUDA, the cartoon filter runs much faster compared to traditional CPU-based execution in Python.  
