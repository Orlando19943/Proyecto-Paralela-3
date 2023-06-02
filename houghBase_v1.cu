/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   :
 To build use  : make
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include "common/pgm.h"

const int degree_increment = 2;
const int total_degree_bins = 180 / degree_increment;
const int total_radial_bins = 100;
const float degree_bin_width = degree_increment * M_PI / 180;

//*****************************************************************
// The CPU function returns a pointer to the accumulator
void CPU_HoughTran(const unsigned char *picture, int width, int height, int **accumulator) {
    // the max radius measured from a corner is the diagonal
    float image_diagonal_length = sqrt(1.0 * width * width + 1.0 * height * height) / 2;
    int x, y;
    float theta;

    *accumulator = new int[total_radial_bins * total_degree_bins];
    memset(*accumulator, 0, sizeof(int) * total_radial_bins * total_degree_bins);

    int x_center = width / 2;
    int y_center = height / 2;
    float radial_bin_width = 2 * image_diagonal_length / total_radial_bins;

    // for each pixel
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            int pixel = j * width + i;
            if (picture[pixel] > 0) {
                // votar por todas las líneas que pasan por ese punto
                x = i - x_center;   // medida desde el centro
                y = y_center - j;   // medida desde el centro invertida
                theta = 0;
                for (int degree_bin = 0; degree_bin < total_degree_bins; degree_bin++) {
                    float radius = x * cos(theta) + y * sin(theta);
                    int radial_bin = (radius + image_diagonal_length) / radial_bin_width;
                    (*accumulator)[radial_bin * total_degree_bins + degree_bin]++; //+1 para este radio radius y este theta
                    theta += degree_bin_width;
                }
            }
        }
    }
}

//*****************************************************************
// TODO usar memoria constante para la tabla de senos y cosenos
// inicializarlo en main y pasarlo al device
//__constant__ float d_Cos[total_degree_bins];
//__constant__ float d_Sin[total_degree_bins];

//*****************************************************************
// TODO Kernel memoria compartida
// __global__ void GPU_HoughTranShared(...)
// {
//   //TODO
// }
// TODO Kernel memoria Constante
// __global__ void GPU_HoughTranConst(...)
// {
//   //TODO
// }

// GPU kernel. One thread per image pixel is spawned.
// The accummulator memory needs to be allocated by the host in global memory
__global__ void GPU_HoughTran(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale, float *d_Cos, float *d_Sin)
{
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;

    if (gloID >= w * h)
        return;

    int xCent = w / 2;
    int yCent = h / 2;

    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    // TODO eventualmente usar memoria compartida para el acumulador

    if (pic[gloID] > 0)
    {
        for (int tIdx = 0; tIdx < total_degree_bins; tIdx++)
        {
            // TODO utilizar memoria constante para senos y cosenos
            // float r = xCoord * cos(tIdx) + yCoord * sin(tIdx); //probar con esto para ver diferencia en tiempo
            float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
            int rIdx = (r + rMax) / rScale;
            // debemos usar atomic, pero que race condition hay si somos un thread por pixel? explique
            atomicAdd(acc + (rIdx * total_degree_bins + tIdx), 1);
        }
    }

    // TODO eventualmente cuando se tenga memoria compartida, copiar del local al global
    // utilizar operaciones atomicas para seguridad
    // faltara sincronizar los hilos del bloque en algunos lados
}

//*****************************************************************
int main(int argc, char **argv) {
    int i;

    if (argc < 2) {
        printf("Usage: %s image\n", argv[0]);
        exit(1);
    }
    printf("Loading image %s\n", argv[1]);

    PGMImage inImg(argv[1]);

    int *cpu_accumulator;
    int width = inImg.x_dim;
    int height = inImg.y_dim;

    printf("Image size is %d x %d\n", width, height);

    float *d_Cos;
    float *d_Sin;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc((void **)&d_Cos, sizeof(float) * total_degree_bins);
    cudaMalloc((void **)&d_Sin, sizeof(float) * total_degree_bins);

    // CPU calculation
    CPU_HoughTran(inImg.pixels, width, height, &cpu_accumulator);

    // pre-compute values to be stored
    float *cosines_values = (float *)malloc(sizeof(float) * total_degree_bins);
    float *sin_values = (float *)malloc(sizeof(float) * total_degree_bins);
    float rad = 0;
    for (i = 0; i < total_degree_bins; i++)
    {
        cosines_values[i] = cos(rad);
        sin_values[i] = sin(rad);
        rad += degree_bin_width;
    }

    float max_radius = sqrt(1.0 * width * width + 1.0 * height * height) / 2;
    float rScale = 2 * max_radius / total_radial_bins;

    // TODO eventualmente volver memoria global
    cudaMemcpy(d_Cos, cosines_values, sizeof(float) * total_degree_bins, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sin, sin_values, sizeof(float) * total_degree_bins, cudaMemcpyHostToDevice);

    // setup and copy data from host to device
    unsigned char *d_in, *h_in;
    int *d_hough, *h_hough;

    h_in = inImg.pixels; // h_in contiene los pixeles de la imagen

    h_hough = (int *)malloc(total_degree_bins * total_radial_bins * sizeof(int));

    cudaMalloc((void **)&d_in, sizeof(unsigned char) * width * height);
    cudaMalloc((void **)&d_hough, sizeof(int) * total_degree_bins * total_radial_bins);
    cudaMemcpy(d_in, h_in, sizeof(unsigned char) * width * height, cudaMemcpyHostToDevice);
    cudaMemset(d_hough, 0, sizeof(int) * total_degree_bins * total_radial_bins);

    // execution configuration uses a 1-D grid of 1-D blocks, each made of 256 threads
    // 1 thread por pixel
    int blockNum = ceil(width * height / 256);
    cudaEventRecord(start);
    GPU_HoughTran<<<blockNum, 256>>>(d_in, width, height, d_hough, max_radius, rScale, d_Cos, d_Sin);
    cudaEventRecord(stop);

    // get results from device
    cudaMemcpy(h_hough, d_hough, sizeof(int) * total_degree_bins * total_radial_bins, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // compare CPU and GPU results
    int count = 0;
    for (i = 0; i < total_degree_bins * total_radial_bins; i++) {
        if (cpu_accumulator[i] != h_hough[i])
            printf("Calculation mismatch at : %i %i %i\n", i, cpu_accumulator[i], h_hough[i]);
        count++;
    }
    printf("Total mismatches: %i\n", count);
    printf("GPU time: %f ms\n", milliseconds);
    printf("Done!\n");

    inImg.to_jpg_with_line("out/test.jpg", cpu_accumulator, 4600, total_degree_bins, degree_increment, total_radial_bins);

    // TODO clean-up
    cudaFree(d_in);
    cudaFree(d_hough);
    free(h_hough);
    free(cpu_accumulator);
    free(cosines_values);
    free(sin_values);

    return 0;
}
