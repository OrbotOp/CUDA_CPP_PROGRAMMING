
//" Addition of two vectors "

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <malloc.h>

__global__ void add(int* a, int* b, int* c, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n)
    {
        c[index] = a[index] + b[index];
    }
}

int main()
{
    int n = 1024;
    int size = n * sizeof(int);

    //memory allocation on the host (CPU)
    int* h_a = (int*)malloc(size);
    int* h_b = (int*)malloc(size);
    int* h_c = (int*)malloc(size);

    //to initialize input arrays, use the loop
    for (int i = 0;i < n;i++)
    {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    //to allocate memory on the device (GPU)
    int* d_a, * d_b, * d_c;

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    //to copy data from the host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    //launch the kernal with n threads
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);

    //copy the output back to the host using 
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    //validate the results
    for (int i = 0;i < n;i++)
    {
        if (h_c[i] != h_a[i] + h_b[i]) {
            printf("Error at index %d\n", i);
            return -1;
        }
    }

    //free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    //free the host memory
    free(h_a);
    free(h_b);
    free(h_c);

    printf("ALL the items are correct!\n");
    return 0;


    return 0;
}
