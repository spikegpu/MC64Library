/* The scattering on device is problematic. */

#include <iostream>
#include <cmath>

#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/gather.h>
#include <thrust/logical.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>

using std::cout;
using std::cerr;
using std::endl;

void old_scatter() {
	double *h_input, *d_input;
	double *h_output, *d_output;
	int    *h_map,  *d_map;

	h_input  = (double *)malloc (sizeof(double) * 10);
	h_output = (double *)malloc (sizeof(double) * 10);
	h_map    = (int *)   malloc (sizeof(int)    * 10);
	cudaMalloc(&d_input, sizeof(double) * 10);
	cudaMalloc(&d_output , sizeof(double) * 10);
	cudaMalloc(&d_map, sizeof(int) * 10);

	h_map[0] = 9;
	h_map[1] = 6;
	h_map[2] = 8;
	h_map[3] = 0;
	h_map[4] = 4;
	h_map[5] = 2;
	h_map[6] = 3;
	h_map[7] = 7;
	h_map[8] = 5;
	h_map[9] = 1;

	for (int i = 0; i < 10; i++)
		h_input[i] = 10.0 + (i + 1);

	cudaMemcpy(d_input, h_input, sizeof(double) * 10, cudaMemcpyHostToDevice);
	cudaMemcpy(d_output, h_output, sizeof(double) * 10, cudaMemcpyHostToDevice);
	cudaMemcpy(d_map,   h_map,   sizeof(int) * 10,    cudaMemcpyHostToDevice);

	//// thrust::scatter(thrust::cuda::par, d_input, d_input + 10, d_map, d_output);
	{
		thrust::device_ptr<double> input_begin(d_input), input_end(d_input + 10), output_begin(d_output);
		thrust::device_ptr<int>    map_begin(d_map);
		thrust::scatter(thrust::cuda::par, input_begin, input_end, map_begin, output_begin);
	}

	cudaMemcpy(h_output, d_output, sizeof(double) * 10, cudaMemcpyDeviceToHost);

	bool correct = true;
	for (int i = 0; i < 10; i++)
		if (h_output[h_map[i]] != h_input[i]) {
			correct = false;
			break;
		}

	if (correct)
		cout << "Old scatter: correct" << endl;
	else
		cout << "Old scatter: INCORRECT" << endl;

	cudaFree(d_map);
	cudaFree(d_input);
	cudaFree(d_output);
	free(h_map);
	free(h_output);
	free(h_input);
}

void old_scatter_if() {
	const double PRESET_VALUE = 10000.0;
	double *h_input, *d_input;
	double *h_output, *d_output;
	int    *h_map,  *d_map;
	bool   *h_stencil, *d_stencil;
	bool   h_output_visited[10] = {0};

	h_input  = (double *)malloc (sizeof(double) * 10);
	h_output = (double *)malloc (sizeof(double) * 10);
	h_map    = (int *)   malloc (sizeof(int)    * 10);
	h_stencil = (bool *)   malloc (sizeof(bool)    * 10);
	cudaMalloc(&d_input, sizeof(double) * 10);
	cudaMalloc(&d_output , sizeof(double) * 10);
	cudaMalloc(&d_map, sizeof(int) * 10);
	cudaMalloc(&d_stencil, sizeof(bool) * 10);

	h_map[0] = 9;
	h_map[1] = 6;
	h_map[2] = 8;
	h_map[3] = 0;
	h_map[4] = 4;
	h_map[5] = 2;
	h_map[6] = 3;
	h_map[7] = 7;
	h_map[8] = 5;
	h_map[9] = 1;

	for (int i = 0; i < 10; i++) {
		h_input[i] = 10.0 + (i + 1);
		h_stencil[i] = ((i & 1) ? true : false);
		h_output[i] = PRESET_VALUE;
	}

	cudaMemcpy(d_input, h_input, sizeof(double) * 10, cudaMemcpyHostToDevice);
	cudaMemcpy(d_output, h_output, sizeof(double) * 10, cudaMemcpyHostToDevice);
	cudaMemcpy(d_stencil, h_stencil, sizeof(bool) * 10, cudaMemcpyHostToDevice);
	cudaMemcpy(d_map,   h_map,   sizeof(int) * 10,    cudaMemcpyHostToDevice);

	//// thrust::scatter_if(thrust::cuda::par, d_input, d_input + 10, d_map, d_stencil, d_output);
	{
		thrust::device_ptr<double> input_begin(d_input), input_end(d_input + 10), output_begin(d_output);
		thrust::device_ptr<int>    map_begin(d_map);
		thrust::device_ptr<bool>   stencil_begin(d_stencil);
		thrust::scatter_if(thrust::cuda::par, input_begin, input_end, map_begin, stencil_begin, output_begin, thrust::identity<bool>());
	}

	cudaMemcpy(h_output, d_output, sizeof(double) * 10, cudaMemcpyDeviceToHost);
		//// thrust::scatter_if(h_input, h_input + 10, h_map, h_stencil, h_output, thrust::identity<bool>());

	bool correct = true;
	for (int i = 0; i < 10; i++)
		if (h_stencil[i]) {
			h_output_visited[h_map[i]] = true;
			if (h_output[h_map[i]] != h_input[i]) {
				correct = false;
				break;
			}
		}

	if (correct) {
		for (int i = 0; i < 10; i++)
			if (!h_output_visited[i]) {
				if (h_output[i] != PRESET_VALUE) {
					correct = false;
					break;
				}
			}
	}

	if (correct)
		cout << "Old scatter_if: correct" << endl;
	else
		cout << "Old scatter_if: INCORRECT" << endl;

	cudaFree(d_map);
	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_stencil);
	free(h_map);
	free(h_output);
	free(h_input);
	free(h_stencil);
}

void new_scatter() {
	double *m_input;
	double *m_output;
	int    *m_map;

	cudaMallocManaged(&m_input, sizeof(double) * 10);
	cudaMallocManaged(&m_output , sizeof(double) * 10);
	cudaMallocManaged(&m_map, sizeof(int) * 10);

	m_map[0] = 9;
	m_map[1] = 6;
	m_map[2] = 8;
	m_map[3] = 0;
	m_map[4] = 4;
	m_map[5] = 2;
	m_map[6] = 3;
	m_map[7] = 7;
	m_map[8] = 5;
	m_map[9] = 1;

	for (int i = 0; i < 10; i++)
		m_input[i] = 10.0 + (i + 1);

	//// thrust::scatter(thrust::cuda::par, m_input, m_input + 10, m_map, m_output);
	{
		thrust::device_ptr<double> input_begin(m_input), input_end(m_input + 10), output_begin(m_output);
		thrust::device_ptr<int>    map_begin(m_map);
		thrust::scatter(thrust::cuda::par, input_begin, input_end, map_begin, output_begin);
	}
	cudaDeviceSynchronize();

	bool correct = true;
	for (int i = 0; i < 10; i++)
		if (m_output[m_map[i]] != m_input[i]) {
			correct = false;
			break;
		}

	if (correct)
		cout << "New scatter: correct" << endl;
	else
		cout << "New scatter: INCORRECT" << endl;

	cudaFree(m_map);
	cudaFree(m_input);
	cudaFree(m_output);
}

void new_scatter_if() {
	const double PRESET_VALUE = 10000.0;
	double *m_input;
	double *m_output;
	int    *m_map;
	bool   *m_stencil;
	bool   h_output_visited[10] = {0};

	cudaMallocManaged(&m_input, sizeof(double) * 10);
	cudaMallocManaged(&m_output , sizeof(double) * 10);
	cudaMallocManaged(&m_map, sizeof(int) * 10);
	cudaMallocManaged(&m_stencil, sizeof(bool) * 10);

	m_map[0] = 9;
	m_map[1] = 6;
	m_map[2] = 8;
	m_map[3] = 0;
	m_map[4] = 4;
	m_map[5] = 2;
	m_map[6] = 3;
	m_map[7] = 7;
	m_map[8] = 5;
	m_map[9] = 1;

	for (int i = 0; i < 10; i++) {
		m_input[i] = 10.0 + (i + 1);
		m_stencil[i] = ((i & 1) ? true : false);
		m_output[i] = PRESET_VALUE;
	}

	//// thrust::scatter_if(thrust::cuda::par, m_input, m_input + 10, m_map, m_stencil, m_output);
	{
		thrust::device_ptr<double> input_begin(m_input), input_end(m_input + 10), output_begin(m_output);
		thrust::device_ptr<int>    map_begin(m_map);
		thrust::device_ptr<bool>   stencil_begin(m_stencil);
		thrust::scatter_if(thrust::cuda::par, input_begin, input_end, map_begin, stencil_begin, output_begin, thrust::identity<bool>());
	}
	cudaDeviceSynchronize();
		//// thrust::scatter_if(h_input, h_input + 10, h_map, h_stencil, h_output, thrust::identity<bool>());

	bool correct = true;
	for (int i = 0; i < 10; i++)
		if (m_stencil[i]) {
			h_output_visited[m_map[i]] = true;
			if (m_output[m_map[i]] != m_input[i]) {
				correct = false;
				break;
			}
		}

	if (correct) {
		for (int i = 0; i < 10; i++)
			if (!h_output_visited[i]) {
				if (m_output[i] != PRESET_VALUE) {
					correct = false;
					break;
				}
			}
	}

	if (correct)
		cout << "New scatter_if: correct" << endl;
	else
		cout << "New scatter_if: INCORRECT" << endl;

	cudaFree(m_map);
	cudaFree(m_input);
	cudaFree(m_output);
	cudaFree(m_stencil);
}

int main(int argc, char **argv) 
{
	old_scatter();
	old_scatter_if();  /* FIXME: this is WRONG on device.*/
	new_scatter();
	new_scatter_if();
	return 0;
}
