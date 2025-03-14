#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <thread>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"


//definitions to video, made them hardcoded, as it was much comfortable for testing
#define INPUT_VIDEO "harry_short.mp4"
#define OUTPUT_VIDEO "harry_become_a_rabbit.mp4"

//=== can be removed, added just for testing
#define TEMP_VIDEO "temp_video_no_audio.mp4"
#define TEMP_AUDIO "temp_audio.aac"
#define MODIFIED_AUDIO "modified_temp_audio.aac"
//====
#define WIDTH 640   
#define HEIGHT 360  

using namespace std;
using namespace std::chrono;

#define BLOCK_SIZE 16  

__global__ void grayscaleKernel(unsigned char*, unsigned char*, int, int);
__global__ void medianBlurKernel(unsigned char*, unsigned char*, int, int);
__global__ void thresholdingKernel(unsigned char*, unsigned char*, int, int);
__global__ void smoothKernel(unsigned char*, unsigned char*, int, int, float, float, float);
__global__ void sharpenKernel(unsigned char*, unsigned char*, int, int);
__global__ void applyEdgeMaskKernel(unsigned char*, unsigned char*, unsigned char*, int, int);


__global__ void grayscaleKernel(unsigned char* input, unsigned char* gray, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        unsigned char r = input[idx];
        unsigned char g = input[idx + 1];
        unsigned char b = input[idx + 2];

        gray[y * width + x] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

__global__ void contrastBoostKernel(unsigned char* input, unsigned char* output, int width, int height, float contrast) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;

        for (int i = 0; i < 3; i++) {
            float pixel = input[idx + i] / 255.0f;
            pixel = (pixel - 0.5f) * contrast + 0.5f;  //contrast formula
            output[idx + i] = min(max(pixel * 255.0f, 0.0f), 255.0f);  
        }
    }
}

__global__ void medianBlurKernel(unsigned char* gray, unsigned char* blurred, int width, int height) {
    __shared__ unsigned char sharedMem[BLOCK_SIZE + 6][BLOCK_SIZE + 6];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int localX = threadIdx.x + 3;  //adding  offset 
    int localY = threadIdx.y + 3;

    //shared memory
    if (x < width && y < height) {
        sharedMem[localY][localX] = gray[y * width + x];

        //border pixels
        if (threadIdx.x < 3) {
            sharedMem[localY][localX - 3] = gray[y * width + max(x - 3, 0)];
            sharedMem[localY][localX + BLOCK_SIZE] = gray[y * width + min(x + BLOCK_SIZE, width - 1)];
        }
        if (threadIdx.y < 3) {
            sharedMem[localY - 3][localX] = gray[max(y - 3, 0) * width + x];
            sharedMem[localY + BLOCK_SIZE][localX] = gray[min(y + BLOCK_SIZE, height - 1) * width + x];
        }
    }
    __syncthreads();

    if (x < width && y < height) {
        unsigned char neighbors[49];
        int k = 0;

        for (int dy = -3; dy <= 3; dy++) {
            for (int dx = -3; dx <= 3; dx++) {
                neighbors[k++] = sharedMem[localY + dy][localX + dx];
            }
        }

        for (int i = 0; i < 49; i++) {
            for (int j = i + 1; j < 49; j++) {
                if (neighbors[j] < neighbors[i]) {
                    unsigned char temp = neighbors[i];
                    neighbors[i] = neighbors[j];
                    neighbors[j] = temp;
                }
            }
        }
        blurred[y * width + x] = neighbors[24];  //median 
    }
}


__global__ void smoothKernel(unsigned char* input, unsigned char* output, int width, int height, float sigmaColor, float sigmaSpace, float smoothingFactor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        float sumR = 0, sumG = 0, sumB = 0, sumWeight = 0;

        for (int dy = -4; dy <= 4; dy++) {
            for (int dx = -4; dx <= 4; dx++) {
                int nx = min(max(x + dx, 0), width - 1);
                int ny = min(max(y + dy, 0), height - 1);
                int nIdx = (ny * width + nx) * 3;

                float colorDist = (input[idx] - input[nIdx]) * (input[idx] - input[nIdx]) +
                    (input[idx + 1] - input[nIdx + 1]) * (input[idx + 1] - input[nIdx + 1]) +
                    (input[idx + 2] - input[nIdx + 2]) * (input[idx + 2] - input[nIdx + 2]);

                float spaceWeight = expf(-((dx * dx + dy * dy) / (2 * sigmaSpace * sigmaSpace)));
                float colorWeight = expf(-(colorDist / (2 * sigmaColor * sigmaColor)));
                float weight = spaceWeight * colorWeight * smoothingFactor;

                sumR += input[nIdx] * weight;
                sumG += input[nIdx + 1] * weight;
                sumB += input[nIdx + 2] * weight;
                sumWeight += weight;
            }
        }

        output[idx] = sumR / sumWeight;
        output[idx + 1] = sumG / sumWeight;
        output[idx + 2] = sumB / sumWeight;
    }
}


__global__ void thresholdingKernel(unsigned char* blurred, unsigned char* edges, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        int sum = 0, count = 0;

        for (int dy = -4; dy <= 4; dy++) {
            for (int dx = -4; dx <= 4; dx++) {
                int nx = min(max(x + dx, 0), width - 1);
                int ny = min(max(y + dy, 0), height - 1);
                sum += blurred[ny * width + nx];
                count++;
            }
        }

        unsigned char threshold = (sum / count) - 6;
        edges[idx] = (blurred[idx] > threshold) ? 255 : 0;
    }
}


__global__ void hsvAdjustmentKernel(unsigned char* input, int width, int height, float saturationBoost, float brightnessBoost, float hueShift) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;

        //rgb 2 hsv
        float r = input[idx] / 255.0f;
        float g = input[idx + 1] / 255.0f;
        float b = input[idx + 2] / 255.0f;

        float maxVal = fmaxf(r, fmaxf(g, b));
        float minVal = fminf(r, fminf(g, b));
        float delta = maxVal - minVal;

        float h = 0.0f, s = 0.0f, v = maxVal;

        if (delta > 0.0001f) {
            s = delta / maxVal;

            if (maxVal == r) {
                h = 60.0f * fmodf((g - b) / delta, 6.0f);
            }
            else if (maxVal == g) {
                h = 60.0f * ((b - r) / delta + 2.0f);
            }
            else {
                h = 60.0f * ((r - g) / delta + 4.0f);
            }
            if (h < 0.0f) h += 360.0f;
        }

        h += hueShift;  //shifting hue to warm colors
        if (h > 360.0f) h -= 360.0f;
        if (h < 0.0f) h += 360.0f;

        s = fminf(fmaxf(s + (saturationBoost / 100.0f), 0.0f), 1.0f);
        v = fminf(fmaxf(v + (brightnessBoost / 100.0f), 0.0f), 1.0f);

        //modify selected colors, for more cartoon image i decided to make it warmer - more yellow and red
        if (h > 300.0f || h < 30.0f) {  //boost red & pink
            s = fminf(s * 1.3f, 1.0f);
            v = fminf(v * 1.1f, 1.0f);
        }
        if (h > 40.0f && h < 70.0f) {  // boost yellow
            s = fminf(s * 1.4f, 1.0f);
        }

        //back to RGB  
        int i = (int)(h / 60.0f) % 6;
        float f = (h / 60.0f) - i;
        float p = v * (1.0f - s);
        float q = v * (1.0f - f * s);
        float t = v * (1.0f - (1.0f - f) * s);

        if (i == 0) { r = v; g = t; b = p; }
        else if (i == 1) { r = q; g = v; b = p; }
        else if (i == 2) { r = p; g = v; b = t; }
        else if (i == 3) { r = p; g = q; b = v; }
        else if (i == 4) { r = t; g = p; b = v; }
        else { r = v; g = p; b = q; }

        // back 2 [0,255]
        input[idx] = (unsigned char)(r * 255.0f);
        input[idx + 1] = (unsigned char)(g * 255.0f);
        input[idx + 2] = (unsigned char)(b * 255.0f);
    }
}


__global__ void sharpenKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;

        int kernel[3][3] = {
            {0, -1, 0},
            {-1, 5, -1},
            {0, -1, 0}
        };


        int sumR = 0, sumG = 0, sumB = 0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = min(max(x + dx, 0), width - 1);
                int ny = min(max(y + dy, 0), height - 1);
                int nIdx = (ny * width + nx) * 3;

                sumR += input[nIdx] * kernel[dy + 1][dx + 1];
                sumG += input[nIdx + 1] * kernel[dy + 1][dx + 1];
                sumB += input[nIdx + 2] * kernel[dy + 1][dx + 1];
            }
        }

        output[idx] = min(max(sumR, 0), 255);
        output[idx + 1] = min(max(sumG, 0), 255);
        output[idx + 2] = min(max(sumB, 0), 255);
    }
}


__global__ void applyEdgeMaskKernel(unsigned char* sharpened, unsigned char* edges, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        int eIdx = y * width + x;

        output[idx] = (edges[eIdx] == 255) ? sharpened[idx] : 0;
        output[idx + 1] = (edges[eIdx] == 255) ? sharpened[idx + 1] : 0;
        output[idx + 2] = (edges[eIdx] == 255) ? sharpened[idx + 2] : 0;

    }
}

#define checkCudaError(val, msg) { if ((val) != cudaSuccess) { \
    std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " (" << msg << "): " \
              << cudaGetErrorString(val) << std::endl; exit(EXIT_FAILURE); } }


void cartoonizeImage(unsigned char* h_input, unsigned char* h_output, int width, int height) {
    unsigned char* d_input, * d_gray, * d_blurred, * d_edges, * d_bilateral, * d_sharpened, * d_output, * d_contr;
    size_t imgSize = width * height * 3;
    size_t graySize = width * height;

    checkCudaError(cudaMalloc(&d_input, imgSize), "Allocating d_input");
    checkCudaError(cudaMalloc(&d_gray, graySize), "Allocating d_gray");
    checkCudaError(cudaMalloc(&d_blurred, graySize), "Allocating d_blurred");
    checkCudaError(cudaMalloc(&d_edges, graySize), "Allocating d_edges");
    checkCudaError(cudaMalloc(&d_bilateral, imgSize), "Allocating d_bilateral");
    checkCudaError(cudaMalloc(&d_sharpened, imgSize), "Allocating d_sharpened");
    checkCudaError(cudaMalloc(&d_output, imgSize), "Allocating d_output");
    checkCudaError(cudaMalloc(&d_contr, imgSize), "Allocating d_contr");

    checkCudaError(cudaMemcpy(d_input, h_input, imgSize, cudaMemcpyHostToDevice), "Copying image to device");

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    //filters
    grayscaleKernel << <grid, block >> > (d_input, d_gray, width, height);
    cudaDeviceSynchronize();

    medianBlurKernel << <grid, block >> > (d_gray, d_blurred, width, height);
    cudaDeviceSynchronize();

    thresholdingKernel << <grid, block >> > (d_blurred, d_edges, width, height);
    cudaDeviceSynchronize();

    smoothKernel << <grid, block >> > (d_input, d_bilateral, width, height, 300.0f, 300.0f, 1.0f);
    cudaDeviceSynchronize();

    contrastBoostKernel << <grid, block >> > (d_bilateral, d_contr, width, height, 1.0f); //you can adjust contrast if you like
    cudaDeviceSynchronize();

    hsvAdjustmentKernel << <grid, block >> > (d_contr, width, height, 25.0f, 30.0f, 5.0f); //three last parameters - saturation, brightness, hue
    cudaDeviceSynchronize();

    smoothKernel << <grid, block >> > (d_contr, d_contr, width, height, 300.0f, 300.0f, 0.001f); //three last parameters - sigmaColor, sigmaSpace, smooth
    cudaDeviceSynchronize();

    sharpenKernel << <grid, block >> > (d_contr, d_sharpened, width, height);
    cudaDeviceSynchronize();

    applyEdgeMaskKernel << <grid, block >> > (d_sharpened, d_edges, d_output, width, height);
    cudaDeviceSynchronize();

    checkCudaError(cudaMemcpy(h_output, d_output, imgSize, cudaMemcpyDeviceToHost), "Copying result to host");

    //free memory
    cudaFree(d_input);
    cudaFree(d_gray);
    cudaFree(d_blurred);
    cudaFree(d_edges);
    cudaFree(d_bilateral);
    cudaFree(d_sharpened);
    cudaFree(d_contr);
    cudaFree(d_output);
}


void processVideo() {
    cout << "========================================================== processing started ================================================================" << endl;
    system("ffmpeg -i " INPUT_VIDEO " -f rawvideo -pix_fmt rgb24 -s 640x360 -y temp_frame.raw");
    ifstream input("temp_frame.raw", ios::binary);
    ofstream output("temp_frame_processed.raw", ios::binary);

    vector<uint8_t> frame(WIDTH * HEIGHT * 3);
    vector<uint8_t> output_frame(WIDTH * HEIGHT * 3);

    while (input.read(reinterpret_cast<char*>(frame.data()), frame.size())) {
        cartoonizeImage(frame.data(), output_frame.data(), WIDTH, HEIGHT);
        output.write(reinterpret_cast<char*>(output_frame.data()), output_frame.size());
    }

    input.close();
    output.close();

    system("ffmpeg -f rawvideo -pix_fmt rgb24 -s 640x360 -r 30 -i temp_frame_processed.raw -c:v libx264 -pix_fmt yuv420p -preset fast -crf 23 -y " TEMP_VIDEO);
    cout << "========================================================== processing finished ================================================================" << endl;
}

void extractAudio() { system("ffmpeg -i " INPUT_VIDEO " -q:a 0 -map a " TEMP_AUDIO " -y"); }
void modifyAudio() { system("ffmpeg -i " TEMP_AUDIO " -filter:a asetrate=44100*1.5,atempo=1/1.5 " MODIFIED_AUDIO " -y"); }
void mergeAudio() { system("ffmpeg -i " TEMP_VIDEO " -i " MODIFIED_AUDIO " -c:v copy -c:a aac " OUTPUT_VIDEO " -y"); }

int main() {
    processVideo();
    extractAudio();
    modifyAudio();
    mergeAudio();
    cout << "Processing completed. Final video: " << OUTPUT_VIDEO << endl;
    return 0;
}
