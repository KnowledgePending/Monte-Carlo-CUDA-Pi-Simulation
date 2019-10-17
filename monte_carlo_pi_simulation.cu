/*
Simple Monte Carlo Pi Simulation using CUDA Primatives
*/

#include <curand.h>
#include <iostream>
#include <iomanip>

__device__ int total_device_points{};

__global__ void measure_points(const float* random_x,
	const float* random_y)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const float x = random_x[i] - 0.5F;
	const float y = random_y[i] - 0.5F;
	const int n = sqrtf(pow(x, 2) + pow(y, 2)) > 0.5F ? 0 : 1;
	atomicAdd(&total_device_points, n);

}
int main() {
	constexpr int width = 512;
	constexpr int height = 512;
	constexpr int count = width * height;
	constexpr int size = count * sizeof(float);

	curandGenerator_t random_generator;

	curandCreateGenerator(&random_generator, CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(random_generator, time(nullptr));

	float *random_x, *random_y;
	cudaMalloc(&random_x, size);
	cudaMalloc(&random_y, size);

	curandGenerateUniform(random_generator, random_x, count);
	curandGenerateUniform(random_generator, random_y, count);

    measure_points << <width, height >> > (random_x, random_y);
    
	int total_host_points;
	cudaMemcpyFromSymbol(&total_host_points, total_device_points, sizeof(int));

	const float estimated_pi = ((4.0F * static_cast<float>(total_host_points)) / static_cast<float>(count));

	std::cout << std::setprecision(std::numeric_limits<float>::digits10 + 1)
		<< "Using the Monte Carlo Method Pi is estimated to be: "
		<< estimated_pi
		<< '\n';

	cudaFree(random_x);
	cudaFree(random_y);
}