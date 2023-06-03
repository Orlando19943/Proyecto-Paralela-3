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
#include <unistd.h>
#include "common/pgm.h"
#include <iostream>
#include <fstream>

const int degree_increment = 2;
const int total_degree_bins = 180 / degree_increment;
const int total_radial_bins = 100;
const int threads_per_block = 256;
const float degree_bin_width = degree_increment * M_PI / 180;

const int total_bins = total_degree_bins * total_radial_bins;

const char *BOLD = "\033[1m";
const char *RED = "\033[91m";
const char *CLEAR = "\033[0m";
const char *GREEN = "\033[92m";

#define START_GPU_TIMING(filename)     \
    cudaEventRecord(start);            \
    file.open(filename, std::ios::app)

#define END_GPU_TIMING(time)                  \
    cudaEventRecord(stop);                    \
    cudaEventSynchronize(stop);               \
    cudaEventElapsedTime(&time, start, stop); \
    file << time << '\n';                     \
    file.close()

/*  ********************************** COMPARE RESULTS ************************************
    * Compara los resultados en memoria de un array en cpu y otro array en memoria CUDA.
    * Imprime las diferencias entre estos y si la diferencia es menor a uno se toma como
    * un error de punto flotante.
    *
    * inputs:
    *  - cpu_results: Puntero a los resultados en CPU.
    *  - in_device_results: Puntero a memoria CUDA con los resultados a comparar
    ***************************************************************************************/
void compare_results(int *cpu_results, int *in_device_results) {
    // copy device memory to local memory for comparison
    int *gpu_results = (int *) malloc(total_bins * sizeof(int));
    cudaMemcpy(gpu_results, in_device_results, sizeof(int) * total_bins, cudaMemcpyDeviceToHost);

    int i;
    int mismatch = 0;
    int rounding_erros = 0;
    for (i = 0; i < total_bins; i++) {
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

/*  ********************************** CPU HOUGH TRANSFORM   ****************************
    * Genera la tabla de votación de la transformada de Hough. Se ejecuta en el CPU.
    *
    * inputs:
    * - picture: Puntero a la imagen a la que se le aplicará la transformada
    * - width: ancho de la imagen
    * - height: altura de la imagen
    *
    * output:
    * - accumulator: Lista con la tabla de votación.
    ***************************************************************************************/
void CPU_HoughTran(const unsigned char *picture, int width, int height, int **accumulator) {
    float image_diagonal_length = sqrt(1.0 * width * width + 1.0 * height * height) / 2;
    int x, y;
    float theta;

    *accumulator = new int[total_bins];
    memset(*accumulator, 0, sizeof(int) * total_bins);

    int x_center = width / 2;
    int y_center = height / 2;
    float radial_bin_width = 2 * image_diagonal_length / total_radial_bins;

    // for each pixel
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            int pixel = j * width + i;
            if (picture[pixel] > 0) {  // si el pixel está marcado
                x = i - x_center;      // medida desde el centro
                y = y_center - j;      // medida desde el centro invertida
                theta = 0;
                // votar por todas las líneas que pasan por ese punto
                for (int degree_bin = 0; degree_bin < total_degree_bins; degree_bin++) {
                    float radius = x * cos(theta) + y * sin(theta);
                    int radial_bin = (radius + image_diagonal_length) / radial_bin_width;
                    int bin = radial_bin * total_degree_bins + degree_bin;
                    (*accumulator)[bin]++;
                    theta += degree_bin_width;
                }
            }
        }
    }
}

/*  ************************* PRECALCULATE TRIGONOMETRY *********************************
    * Precalcula los senos y cosenos de los diferentes bins de los ángulos.
    *
    * También llena la memoria global const_cos y const_sin con los resultados.
    *
    * outputs:
    * - device_cosine: Un puntero a una lista en memoria cuda con los cos precalculados
    * - device_sine: Un puntero a una lista en memoria cuda con los sin precalculados
    ***************************************************************************************/
__device__ __constant__ float const_cos[total_degree_bins];
__device__ __constant__ float const_sin[total_degree_bins];

void precalculate_trigonometry(float **device_cosine, float **device_sin) {
    // allocate cpu memory for results
    float *precomputed_cos = (float *) malloc(sizeof(float) * total_degree_bins);
    float *precomputed_sin = (float *) malloc(sizeof(float) * total_degree_bins);

    // fill cpu precomputed values
    int i;
    float degree = 0;
    for (i = 0; i < total_degree_bins; i++) {
        precomputed_cos[i] = cos(degree);
        precomputed_sin[i] = sin(degree);

        degree += degree_bin_width;
    }

    // move local to global const memory
    cudaMemcpyToSymbol(const_cos, precomputed_cos, sizeof(float) * total_degree_bins);
    cudaMemcpyToSymbol(const_sin, precomputed_sin, sizeof(float) * total_degree_bins);

    // move to device memory
    cudaMalloc((void **) device_cosine, sizeof(float) * total_degree_bins);
    cudaMalloc((void **) device_sin, sizeof(float) * total_degree_bins);

    // free mem
    cudaMemcpy(*device_cosine, precomputed_cos, sizeof(float) * total_degree_bins, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_sin, precomputed_sin, sizeof(float) * total_degree_bins, cudaMemcpyHostToDevice);
    free(precomputed_cos);
    free(precomputed_sin);
}


/*  ****************************** GPU HOUGH TRANS SHARED *********************************
    * Precalcula la transformada de Hough en la GPU usando memoria compartida, usando cos
    * y sin precalculados en memoria constante.
    *
    * inputs:
    * - picture: Puntero de CUDA con la información de la imagen
    * - width: Ancho de la imagen
    * - height: Altura de la imagen
    * - image_diagonal_length: la diagonal de la imagen precalculada
    * - radial_bin_width: el ancho de cada casilla de la tabla en el eje del radio precalculado
    *
    * outputs:
    * - accumulator: La lista de votación en CUDA llena.
    ***************************************************************************************/
__global__ void GPU_HoughTranShared(
        unsigned char *picture,
        int width, int height,
        int *accumulator,
        float image_diagonal_length,
        float radial_bin_width
) {
    __shared__ int partial_accumulator[total_bins];
    int local_id = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // set who's responsible for initializing and moving a bin to global
    int bins_per_thread = total_bins / threads_per_block;
    int bins_start = local_id * bins_per_thread;
    int bins_end = (local_id + 1) * bins_per_thread;
    // assign remainder bins to the last thread
    if (local_id == threads_per_block - 1) {
        bins_end = total_degree_bins * total_radial_bins;
    }

    // initialize memory to zero
    for (int i = bins_start; i < bins_end; i++) {
        partial_accumulator[i] = 0;
    }

    __syncthreads();


    if (global_id >= width * height)
        return;

    int x_center = width / 2;
    int y_center = height / 2;

    int x = global_id % width - x_center;
    int y = y_center - global_id / width;

    // fill local poll table
    if (picture[global_id] > 0) {
        for (int bin_degree = 0; bin_degree < total_degree_bins; bin_degree++) {
            float radius = x * const_cos[bin_degree] + y * const_sin[bin_degree];
            int radial_bin = (radius + image_diagonal_length) / radial_bin_width;
            atomicAdd_block(partial_accumulator + (radial_bin * total_degree_bins + bin_degree), 1);
        }
    }

    __syncthreads();

    // update global poll table
    for (int i = bins_start; i < bins_end; i++) {
        if (partial_accumulator[i] > 0) {
            atomicAdd(accumulator + i, partial_accumulator[i]);
        }
    }
}


/*  ****************************** GPU HOUGH TRANS CONST **********************************
    * Precalcula la transformada de Hough en la GPU sin memoria compartida, usando cos
    * y sin precalculados en memoria constante.
    *
    * inputs:
    * - picture: Puntero de CUDA con la información de la imagen
    * - width: Ancho de la imagen
    * - height: Altura de la imagen
    * - image_diagonal_length: la diagonal de la imagen precalculada
    * - radial_bin_width: el ancho de cada casilla de la tabla en el eje del radio precalculado
    *
    * outputs:
    * - accumulator: La lista de votación en CUDA llena.
    ***************************************************************************************/
__global__ void GPU_HoughTranConst(
        unsigned char *picture,
        int width,
        int height,
        int *accumulator,
        float image_diagonal_length,
        float radial_bin_width
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_id >= width * height)
        return;

    int x_center = width / 2;
    int y_center = height / 2;

    int x = global_id % width - x_center;
    int y = y_center - global_id / width;

    if (picture[global_id] > 0) {
        for (int bin_degree = 0; bin_degree < total_degree_bins; bin_degree++) {
            float radius = x * const_cos[bin_degree] + y * const_sin[bin_degree];
            int radial_bin = (radius + image_diagonal_length) / radial_bin_width;
            atomicAdd(accumulator + (radial_bin * total_degree_bins + bin_degree), 1);
        }
    }
}

/*  ************************ GPU HOUGH TRANS PRECOMPUTED *********************************
    * Precalcula la transformada de Hough en la GPU sin memoria compartida, usando cos
    * y sin precalculados como parámetro.
    *
    * inputs:
    * - picture: Puntero de CUDA con la información de la imagen
    * - width: Ancho de la imagen
    * - height: Altura de la imagen
    * - image_diagonal_length: la diagonal de la imagen precalculada
    * - radial_bin_width: el ancho de cada casilla de la tabla en el eje del radio precalculado
    * - precomputed_cos: lista con todos los posibles valores de cos precalculados
    * - precomputed_sin: lista con todos los posibles valores de sin precalculados
    *
    * outputs:
    * - accumulator: La lista de votación en CUDA llena.
    ***************************************************************************************/
__global__ void GPU_HoughTranPreComputed(
        unsigned char *picture,
        int width,
        int height,
        int *accumulator,
        float image_diagonal_length,
        float radial_bin_width,
        float *precomputed_cos,
        float *precomputed_sin
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_id >= width * height)
        return;

    int x_center = width / 2;
    int y_center = height / 2;

    int x = global_id % width - x_center;
    int y = y_center - global_id / width;

    if (picture[global_id] > 0) {
        for (int bin_degree = 0; bin_degree < total_degree_bins; bin_degree++) {
            float radius = x * precomputed_cos[bin_degree] + y * precomputed_sin[bin_degree];
            int radial_bin = (radius + image_diagonal_length) / radial_bin_width;
            atomicAdd(accumulator + (radial_bin * total_degree_bins + bin_degree), 1);
        }
    }
}

/*  ****************************** GPU HOUGH TRANS ****************************************
    * Precalcula la transformada de Hough en la GPU sin ningún tipo de optimización
    *
    * inputs:
    * - picture: Puntero de CUDA con la información de la imagen
    * - width: Ancho de la imagen
    * - height: Altura de la imagen
    * - image_diagonal_length: la diagonal de la imagen precalculada
    * - radial_bin_width: el ancho de cada casilla de la tabla en el eje del radio precalculado
    *
    * outputs:
    * - accumulator: La lista de votación en CUDA llena.
    ***************************************************************************************/
__global__ void GPU_HoughTran(
        unsigned char *picture,
        int width,
        int height,
        int *accumulator,
        float image_diagonal_length,
        float radial_bin_width
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_id >= width * height)
        return;

    int x_center = width / 2;
    int y_center = height / 2;

    int x = global_id % width - x_center;
    int y = y_center - global_id / width;

    if (picture[global_id] > 0) {
        for (int bin_degree = 0; bin_degree < total_degree_bins; bin_degree++) {
             float radius = x * cos(bin_degree) + y * sin(bin_degree);
            int radial_bin = (radius + image_diagonal_length) / radial_bin_width;
            atomicAdd(accumulator + (radial_bin * total_degree_bins + bin_degree), 1);
        }
    }
}

//  ***************************************************************************************
//  ************************************** START MAIN *************************************
//  ***************************************************************************************
int main(int argc, char **argv) {
    // get input from user
    if (argc < 2) {
        printf("Usage: %s image\n", argv[0]);
        exit(1);
    }
    printf("Loading image %s\n", argv[1]);

    // write image to mem
    PGMImage inImg(argv[1]);
    int width = inImg.x_dim;
    int height = inImg.y_dim;

    printf("Image size is %d x %d\n", width, height);

    // write the image to device
    unsigned char *image_in_device;
    cudaMalloc((void **) &image_in_device, sizeof(unsigned char) * width * height);
    cudaMemcpy(image_in_device, inImg.pixels, sizeof(unsigned char) * width * height, cudaMemcpyHostToDevice);

    // pre-compute values to be stored
    int blockNum = ceil(width * height / threads_per_block);

    float *precomputed_cos, *precomputed_sin;
    precalculate_trigonometry(&precomputed_cos, &precomputed_sin);

    float max_radius = sqrt(1.0 * width * width + 1.0 * height * height) / 2;
    float radial_bin_width = 2 * max_radius / total_radial_bins;

    //  =======================================================================================
    //  ============================ START ACTUAL CALCULATIONS ================================
    //  =======================================================================================
    std::ofstream file;

    int *device_accumulator;
    cudaMalloc((void **) &device_accumulator, sizeof(int) * total_degree_bins * total_radial_bins);

    // for time measuring
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // CPU calculation
    int *cpu_accumulator;
    CPU_HoughTran(inImg.pixels, width, height, &cpu_accumulator);

    // ------------------ GPU - No Shared Memory Nor constant memory -----------------
    printf("\n%s%sGPU - Sin memoria compartida ni constante %s\n", BOLD, RED, CLEAR);
    cudaMemset(device_accumulator, 0, sizeof(int) * total_bins);

    START_GPU_TIMING("out/gpu_hough_trans.txt");
    GPU_HoughTran<<<blockNum, threads_per_block>>>(
            image_in_device,
            width,
            height,
            device_accumulator,
            max_radius,
            radial_bin_width
    );
    END_GPU_TIMING(milliseconds);

    compare_results(cpu_accumulator, device_accumulator);
    printf("%sGPU time: %f ms%s\n", GREEN, milliseconds, CLEAR);

    // ------------------ GPU - No Shared Memory Nor constant memory Cos and Sin PreComputed -----------------
    printf("\n%s%sGPU - Sin memoria compartida ni constante. Cos y Sin precomputados %s\n", BOLD, RED, CLEAR);
    cudaMemset(device_accumulator, 0, sizeof(int) * total_bins);

    START_GPU_TIMING("out/gpu_hough_trans_precomputed.txt");
    GPU_HoughTranPreComputed<<<blockNum, threads_per_block>>>(
            image_in_device,
            width,
            height,
            device_accumulator,
            max_radius,
            radial_bin_width,
            precomputed_cos,
            precomputed_sin
    );
    END_GPU_TIMING(milliseconds);

    compare_results(cpu_accumulator, device_accumulator);
    printf("%sGPU time: %f ms%s\n", GREEN, milliseconds, CLEAR);


    // ------------------------------ GPU - const memory ------------------------------
    printf("\n%s%sGPU - Sin memoria compartida, pero con memoria constante %s\n", BOLD, RED, CLEAR);
    cudaMemset(device_accumulator, 0, sizeof(int) * total_bins);

    START_GPU_TIMING("out/gpu_hough_trans_const.txt");
    GPU_HoughTranConst<<<blockNum, threads_per_block>>>(
            image_in_device,
            width,
            height,
            device_accumulator,
            max_radius,
            radial_bin_width
    );
    END_GPU_TIMING(milliseconds);

    compare_results(cpu_accumulator, device_accumulator);
    printf("%sGPU time: %f ms%s\n", GREEN, milliseconds, CLEAR);

    // --------------------- GPU - const memory and shared memory --------------------
    printf("\n%s%sGPU - Memoria compartida y memoria constante %s\n", BOLD, RED, CLEAR);
    cudaMemset(device_accumulator, 0, sizeof(int) * total_bins);

    START_GPU_TIMING("out/gpu_hough_trans_const_shared.txt");
    GPU_HoughTranShared<<<blockNum, threads_per_block>>>(
            image_in_device,
            width,
            height,
            device_accumulator,
            max_radius,
            radial_bin_width
    );
    END_GPU_TIMING(milliseconds);

    compare_results(cpu_accumulator, device_accumulator);
    printf("%sGPU time: %f ms%s\n", GREEN, milliseconds, CLEAR);

    //  =======================================================================================
    //  ================================= END OF CALCULATIONS ================================
    //  =======================================================================================

    // save results
    inImg.to_jpg_with_line(
            "out/test.jpg",
            cpu_accumulator,
            4300,
            total_degree_bins,
            degree_increment,
            total_radial_bins
    );

    // cleanup
    free(cpu_accumulator);
    cudaFree(image_in_device);
    cudaFree(device_accumulator);
    cudaFree(precomputed_sin);
    cudaFree(precomputed_cos);

    return 0;
}
