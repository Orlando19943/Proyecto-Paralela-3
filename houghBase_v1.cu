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
const int threads_per_block = 32;
const float degree_bin_width = degree_increment * M_PI / 180;

const char* BOLD = "\033[1m";
const char* RED = "\033[91m";
const char* CLEAR = "\033[0m";
const char* GREEN = "\033[92m";

void compare_results(int* cpu_results, int* in_device_results) {
    int *gpu_results = (int *) malloc(total_degree_bins * total_radial_bins * sizeof(int));
    cudaMemcpy(gpu_results, in_device_results, sizeof(int) * total_degree_bins * total_radial_bins, cudaMemcpyDeviceToHost);

    int i;
    int mismatch = 0;
    int rounding_erros = 0;
    for (i = 0; i < total_degree_bins * total_radial_bins; i++) {
        if (cpu_results[i] != gpu_results[i]) {
            if (gpu_results[i] - 1 == cpu_results[i] || gpu_results[i] + 1 == cpu_results[i]) {
                rounding_erros += 1;
            } else {
                printf(" - Calculation mismatch at : %i %i %i\n", i, cpu_results[i], gpu_results[i]);
                mismatch++;
            }
        }
    }

    free(gpu_results);
    printf("Total possible rounding errors (±1): %i\n", rounding_erros);
    printf("Total mismatches: %i\n", mismatch);
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
__device__ __constant__ float pre_cosine[total_degree_bins];
__device__ __constant__ float pre_sin[total_degree_bins];

void precalculate_trigonometry(float **device_cosine, float **device_sin) {
    float *precomputed_cos = (float *)malloc(sizeof(float) * total_degree_bins);
    float *precomputed_sin = (float *)malloc(sizeof(float) * total_degree_bins);

    int i;
    float degree = 0;
    for (i = 0; i < total_degree_bins; i++) {
        precomputed_cos[i] = cos(degree);
        precomputed_sin[i] = sin(degree);

        degree += degree_bin_width;
    }

    // fill global const mem
    cudaMemcpyToSymbol(pre_cosine, precomputed_cos, sizeof(float) * total_degree_bins);
    cudaMemcpyToSymbol(pre_sin, precomputed_sin, sizeof(float) * total_degree_bins);

    // move to device
    cudaMalloc((void **)device_cosine, sizeof(float) * total_degree_bins);
    cudaMalloc((void **)device_sin, sizeof(float) * total_degree_bins);

    cudaMemcpy(*device_cosine, precomputed_cos, sizeof(float) * total_degree_bins, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_sin, precomputed_sin, sizeof(float) * total_degree_bins, cudaMemcpyHostToDevice);

    free(precomputed_cos);
    free(precomputed_sin);
}


//*****************************************************************
__global__ void GPU_HoughTranShared(unsigned char *picture, int width, int height, int *accumulator, float image_diagonal_length, float radial_bin_width) {
    __shared__ int partial_accumulator[total_degree_bins * total_radial_bins];

    int local_id = threadIdx.x;
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
            atomicAdd(&partial_accumulator[radial_bin * total_degree_bins + bin_degree], 1);
        }
    }

    // sync cosos
    __syncthreads();

    if (local_id == 0) {
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                int pixel = j * width + i;
                if (partial_accumulator[pixel] > 0) {
                    atomicAdd(&partial_accumulator[pixel], partial_accumulator[pixel]);
                }
            }
        }
        printf("YAA TERMINEEE");
    }
    __syncthreads();
}

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
    if (argc < 2) {
        printf("Usage: %s image\n", argv[0]);
        exit(1);
    }
    printf("Loading image %s\n", argv[1]);

    // write image to mem
    PGMImage inImg(argv[1]);
    int width = inImg.x_dim;
    int height = inImg.y_dim;

    // also write the image to device
    unsigned char *image_in_device;
    cudaMalloc((void **)&image_in_device, sizeof(unsigned char) * width * height);
    cudaMemcpy(image_in_device, inImg.pixels, sizeof(unsigned char) * width * height, cudaMemcpyHostToDevice);

    printf("Image size is %d x %d\n", width, height);

    // pre-compute values to be stored
    int blockNum = ceil(width * height / threads_per_block);

    float *precomputed_cos, *precomputed_sin;
    precalculate_trigonometry(&precomputed_cos, &precomputed_sin);

    float max_radius = sqrt(1.0 * width * width + 1.0 * height * height) / 2;
    float radial_bin_width = 2 * max_radius / total_radial_bins;

    //  ========== START ACTUAL CALCULATIONS ==================
    int *device_accumulator;
    cudaMalloc((void **)&device_accumulator, sizeof(int) * total_degree_bins * total_radial_bins);

    // for time measuring
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // CPU calculation
    int *cpu_accumulator;
    CPU_HoughTran(inImg.pixels, width, height, &cpu_accumulator);

    // ================================================================
    // GPU - No Shared Memory nor constant calculation
    printf("\n%s%sGPU - No const nor shared mem%s\n", BOLD, RED, CLEAR);
    cudaMemset(device_accumulator, 0, sizeof(int) * total_degree_bins * total_radial_bins);

//    cudaEventRecord(start);
//    GPU_HoughTran<<<blockNum, threads_per_block>>>(image_in_device, width, height, device_accumulator, max_radius, radial_bin_width, precomputed_cos, precomputed_sin);
//    cudaEventRecord(stop);
//    cudaEventSynchronize(stop);
//    cudaEventElapsedTime(&milliseconds, start, stop);
//
//    compare_results(cpu_accumulator, device_accumulator);
//    printf("%sGPU time: %f ms%s\n", GREEN, milliseconds, CLEAR);

    // ================================================================
    // GPU - Const Memory Only
    printf("\n%s%sGPU - No const nor shared mem%s\n", BOLD, RED, CLEAR);
    cudaMemset(device_accumulator, 0, sizeof(int) * total_degree_bins * total_radial_bins);

    cudaEventRecord(start);
    GPU_HoughTranConst<<<blockNum, threads_per_block>>>(image_in_device, width, height, device_accumulator, max_radius, radial_bin_width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    compare_results(cpu_accumulator, device_accumulator);
    printf("%sGPU time: %f ms%s\n", GREEN, milliseconds, CLEAR);

    // ================================================================
    // GPU - Shared Memory const
    printf("\n%s%sGPU - No const nor shared mem%s\n", BOLD, RED, CLEAR);
    cudaMemset(device_accumulator, 0, sizeof(int) * total_degree_bins * total_radial_bins);


//    cudaEventRecord(start);
//    GPU_HoughTranShared<<<blockNum, threads_per_block>>>(image_in_device, width, height, device_accumulator, max_radius, radial_bin_width);
//    cudaEventRecord(stop);
//    cudaEventSynchronize(stop);
//    cudaEventElapsedTime(&milliseconds, start, stop2;

    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2);
    GPU_HoughTranShared<<<blockNum, threads_per_block>>>(image_in_device, width, height, device_accumulator, max_radius, radial_bin_width);
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&milliseconds, start2, stop2);

    compare_results(cpu_accumulator, device_accumulator);
    printf("%sGPU time: %f ms%s\n", GREEN, milliseconds, CLEAR);


    // save results
    inImg.to_jpg_with_line("out/test.jpg", cpu_accumulator, 4600, total_degree_bins, degree_increment, total_radial_bins);

    // cleanup
    free(cpu_accumulator);
    cudaFree(image_in_device);
    cudaFree(device_accumulator);
    cudaFree(precomputed_sin);
    cudaFree(precomputed_cos);

    return 0;
}
