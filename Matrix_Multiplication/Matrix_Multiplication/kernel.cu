//"
// CUDA kernel funtion for matrix multiplication.
// A, B are the input matrices, and C is the result matrix.
// N is the dimension of the square matrix.
//"

#include <iostream>
#include <cuda_runtime.h>
#include <malloc.h>

__global__ void MatMul(const int* A, const int* B, int* C, int N)
{
	//calculate the row and column index for the current thread
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	//Make sure the thread is within the bounds of the matrix dimensions
	if (row < N && col < N)
	{
		int sum = 0;
		for (int i = 0;i < N;i++)
		{
			sum += A[row * N + i] * B[i * N + col];
		}

		//store the rsult in matrix C
		C[row * N + col] = sum;
	}
}

int main()
{	
	//define the size of the NxN matrix
	int N;

	// Ask the user to input the matrix size
	std::cout << "Enter the dimension of the square matrix (N): ";
	std::cin >> N;
	
	
	int size = N * N * sizeof(int); // calculate the memory size required for each matrix

	//ALlocate memory on the host (CPU) for matrices A, B, and C
	int* h_a = (int*)malloc(size);
	int* h_b = (int*)malloc(size);
	int* h_c = (int*)malloc(size);
	

	// Allow the user to input values for matrices A and B
	std::cout << "Enter the elements of matrix A (row-wise):\n";
	for (int i = 0; i < N * N; i++) {
		std::cin >> h_a[i];
	}

	std::cout << "Enter the elements of matrix B (row-wise):\n";
	for (int i = 0; i < N * N; i++) {
		std::cin >> h_b[i];
	}

	//Allocate memory on the devices (GPU) for matrices A, B, C
	int* d_a, * d_b, * d_c;
	
	cudaMalloc(&d_a, size);
	cudaMalloc(&d_b, size);
	cudaMalloc(&d_c, size);


	// copy matrices A and B from the host (CPU) to device (GPU)
	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

	// Create CUDA events for timing
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Record the start event
	cudaEventRecord(start, 0);

	//Define the numbers of threads per block in both dimentions (16x16 threads per block)
	dim3 threadsPerBlock(16, 16);

	//calculate the number of blocks needed in each dimentions

	dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
						(N +threadsPerBlock.y - 1) / threadsPerBlock.y);

	//Launch the MatMul kernel on GPU
	MatMul << <blocksPerGrid, threadsPerBlock >> > (d_a,d_b,d_c,N);


	// Record the stop event
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop); // Wait for the event to complete

	// Calculate the elapsed time in milliseconds
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	//copy result from device(GPU) to host(CPU)
	cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

	// Output the result matrix
	std::cout << "Resulting matrix C (after multiplication):\n";
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			std::cout << h_c[i * N + j] << " ";
		}
		std::cout << std::endl;
	}

	// Output the performance result
	std::cout << "Matrix multiplication took " << milliseconds << " milliseconds on the GPU." << std::endl;


	//free the memory allocated to device
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	//free the memory allocated to host
	free(h_a);
	free(h_b);
	free(h_c);

	std::cout << "MAtrix multiplication complete!!" << std::endl;

	return 0;
}