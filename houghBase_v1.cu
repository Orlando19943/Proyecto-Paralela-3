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

void compare_results(int* cpu_results, int* in_device_results) {
    int *gpu_results = (int *) malloc(total_degree_bins * total_radial_bins * sizeof(int));
    cudaMemcpy(gpu_results, in_device_results, sizeof(int) * total_degree_bins * total_radial_bins, cudaMemcpyDeviceToHost);

    int i;
    int count = 0;
    for (i = 0; i < total_degree_bins * total_radial_bins; i++) {
        if (cpu_results[i] != gpu_results[i]) {
            printf("Calculation mismatch at : %i %i %i\n", i, cpu_results[i], gpu_results[i]);
            count++;
        }
    }

    free(gpu_results);
    printf("Total mismatches: %i\n", count);
}

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
__constant__ float pre_cosine[total_degree_bins];
__constant__ float pre_sin[total_degree_bins];

void precalculate_trigonometry(float **device_cosine, float **device_sin) {
    float *precomputed_cos = (float *)malloc(sizeof(float) * total_degree_bins);
    float *precomputed_sin = (float *)malloc(sizeof(float) * total_degree_bins);

    int i;
    float degree = 0;
    for (i = 0; i < total_degree_bins; i++) {
        float cosine = cos(degree);
        float sine = sin(degree);

        // fill both cosine/sin tables
        precomputed_cos[i] = cosine;
        precomputed_sin[i] = sine;
        pre_cosine[i] = cosine;
        pre_sin[i] = cosine;

        degree += degree_bin_width;
    }

    // move to device
    cudaMalloc((void **)device_cosine, sizeof(float) * total_degree_bins);
    cudaMalloc((void **)device_sin, sizeof(float) * total_degree_bins);

    cudaMemcpy(*device_cosine, precomputed_cos, sizeof(float) * total_degree_bins, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_sin, precomputed_sin, sizeof(float) * total_degree_bins, cudaMemcpyHostToDevice);

    free(precomputed_cos);
    free(precomputed_sin);
}


//*****************************************************************
// TODO Kernel memoria compartida
// __global__ void GPU_HoughTranShared(...)
// {
//   //TODO
// }

__global__ void GPU_HoughTranConst(unsigned char *picture, int width, int height, int *accumulator, float image_diagonal_length, float radial_bin_width) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_id >= width * height)
        return;

    int x_center = width / 2;
    int y_center = height / 2;

    int x = global_id % width - x_center;
    int y = y_center - global_id / width;

    if (picture[global_id] > 0) {
        for (int bin_degree = 0; bin_degree < total_degree_bins; bin_degree++) {
            float radius = x * pre_cosine[bin_degree] + y * pre_sin[bin_degree];
            int radial_bin = (radius + image_diagonal_length) / radial_bin_width;
            atomicAdd(accumulator + (radial_bin * total_degree_bins + bin_degree), 1);
        }
    }

    // TODO eventualmente cuando se tenga memoria compartida, copiar del local al global
    // utilizar operaciones atomicas para seguridad
    // faltara sincronizar los hilos del bloque en algunos lados
}

// GPU kernel. One thread per image pixel is spawned.
// The accummulator memory needs to be allocated by the host in global memory
__global__ void GPU_HoughTran(unsigned char *picture, int width, int height, int *accumulator, float image_diagonal_length, float radial_bin_width, float *precomputed_cos, float *precomputed_sin) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_id >= width * height)
        return;

    int x_center = width / 2;
    int y_center = height / 2;

    int x = global_id % width - x_center;
    int y = y_center - global_id / width;

    if (picture[global_id] > 0) {
        for (int bin_degree = 0; bin_degree < total_degree_bins; bin_degree++) {
            // float radius = x * cos(bin_degree) + y * sin(bin_degree); //probar con esto para ver diferencia en tiempo
            float radius = x * precomputed_cos[bin_degree] + y * precomputed_sin[bin_degree];
            int radial_bin = (radius + image_diagonal_length) / radial_bin_width;
            // debemos usar atomic, pero que race condition hay si somos un thread por pixel? explique
            atomicAdd(accumulator + (radial_bin * total_degree_bins + bin_degree), 1);
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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // CPU calculation
    CPU_HoughTran(inImg.pixels, width, height, &cpu_accumulator);

    // pre-compute values to be stored
    float *precomputed_cos, *precomputed_sin;
    precalculate_trigonometry(&precomputed_cos, &precomputed_sin);

    float max_radius = sqrt(1.0 * width * width + 1.0 * height * height) / 2;
    float radial_bin_width = 2 * max_radius / total_radial_bins;

    // setup and copy data from host to device
    unsigned char *image_in_device;
    int *d_hough;

    cudaMalloc((void **)&image_in_device, sizeof(unsigned char) * width * height);
    cudaMalloc((void **)&d_hough, sizeof(int) * total_degree_bins * total_radial_bins);
    cudaMemcpy(image_in_device, inImg.pixels, sizeof(unsigned char) * width * height, cudaMemcpyHostToDevice);
    cudaMemset(d_hough, 0, sizeof(int) * total_degree_bins * total_radial_bins);

    // execution configuration uses a 1-D grid of 1-D blocks, each made of 256 threads
    // 1 thread por pixel
    int blockNum = ceil(width * height / 256);
    cudaEventRecord(start);
    GPU_HoughTran<<<blockNum, 256>>>(image_in_device, width, height, d_hough, max_radius, radial_bin_width, precomputed_cos, precomputed_sin);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // compare CPU and GPU results
    compare_results(cpu_accumulator, d_hough);
    printf("GPU time: %f ms\n", milliseconds);
    printf("Done!\n");

    inImg.to_jpg_with_line("out/test.jpg", cpu_accumulator, 4600, total_degree_bins, degree_increment, total_radial_bins);

    cudaFree(image_in_device);
    cudaFree(d_hough);
    free(cpu_accumulator);
    cudaFree(precomputed_sin);
    cudaFree(precomputed_cos);

    return 0;
}
