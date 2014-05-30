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

void old_fill() {
	const int ARRAY_SIZE = 1000;

	double *hA, *dA;

	hA = (double *)malloc(sizeof(double) * ARRAY_SIZE);
	cudaMalloc(&dA, sizeof(double) * ARRAY_SIZE);

	for (int i = 0; i < ARRAY_SIZE; i++)
		hA[i] = 0.0;

	thrust::fill(thrust::cuda::par, dA, dA + ARRAY_SIZE, 9.0);
	cudaMemcpy(hA, dA, sizeof(double) * ARRAY_SIZE, cudaMemcpyDeviceToHost);

	bool correct = true;
	for (int i = 0; i < ARRAY_SIZE; i++)
		if (hA[i] != 9.0) {
			correct = false;
			break;
		}

	if (correct)
		cout << "Old fill: correct" << endl;
	else
		cout << "Old fill: INCORRECT" << endl;

	free(hA);
	cudaFree(dA);
}

void new_fill() {
	const int ARRAY_SIZE = 1000;

	double *mA;

	cudaMallocManaged(&mA, sizeof(double) * ARRAY_SIZE);

	for (int i = 0; i < ARRAY_SIZE; i++)
		mA[i] = 0.0;

	thrust::fill(thrust::cuda::par, mA, mA + ARRAY_SIZE, 9.0);
	cudaDeviceSynchronize();

	bool correct = true;
	for (int i = 0; i < ARRAY_SIZE; i++)
		if (mA[i] != 9.0) {
			correct = false;
			break;
		}

	if (correct)
		cout << "New fill: correct" << endl;
	else
		cout << "New fill: INCORRECT" << endl;

	cudaFree(mA);
}

void old_copy() {
	const int ARRAY_SIZE = 1000;

	double *hA, *dA, *dB, *hB;

	hA = (double *)malloc(sizeof(double) * ARRAY_SIZE);
	hB = (double *)malloc(sizeof(double) * ARRAY_SIZE);
	cudaMalloc(&dA, sizeof(double) * ARRAY_SIZE);
	cudaMalloc(&dB, sizeof(double) * ARRAY_SIZE);

	for (int i = 0; i < ARRAY_SIZE; i++)
		hA[i] = 1.0 * (i+1);

	cudaMemcpy(dA, hA, sizeof(double) * ARRAY_SIZE, cudaMemcpyHostToDevice);
	thrust::copy(thrust::cuda::par, dA, dA + ARRAY_SIZE, dB);
	cudaMemcpy(hB, dB, sizeof(double) * ARRAY_SIZE, cudaMemcpyDeviceToHost);

	bool correct = true;
	for (int i = 0; i < ARRAY_SIZE; i++)
		if (hB[i] != 1.0 * (i + 1)) {
			correct = false;
			break;
		}

	if (correct)
		cout << "Old copy: correct" << endl;
	else
		cout << "Old copy: INCORRECT" << endl;

	free(hA);
	cudaFree(dA);
	free(hB);
	cudaFree(dB);
}

void new_copy() {
	const int ARRAY_SIZE = 1000;

	double *mA, *mB;

	cudaMallocManaged(&mA, sizeof(double) * ARRAY_SIZE);
	cudaMallocManaged(&mB, sizeof(double) * ARRAY_SIZE);

	for (int i = 0; i < ARRAY_SIZE; i++)
		mA[i] = 1.0 * (i+1);

	thrust::copy(thrust::cuda::par, mA, mA + ARRAY_SIZE, mB);
	cudaDeviceSynchronize();

	bool correct = true;
	for (int i = 0; i < ARRAY_SIZE; i++)
		if (mB[i] != 1.0 * (i + 1)) {
			correct = false;
			break;
		}

	if (correct)
		cout << "New copy: correct" << endl;
	else
		cout << "New copy: INCORRECT" << endl;

	cudaFree(mA);
	cudaFree(mB);
}

void old_sequence() {
	const int ARRAY_SIZE = 1000;

	double *hA, *dA;

	hA = (double *)malloc(sizeof(double) * ARRAY_SIZE);
	cudaMalloc(&dA, sizeof(double) * ARRAY_SIZE);

	for (int i = 0; i < ARRAY_SIZE; i++)
		hA[i] = 0.0;

	/* FIXME: it's not correct to call
	    thrust::sequence(thrust::cuda::par, dA, dA + ARRAY_SIZE);
	   */
	{
		thrust::device_ptr<double> A_begin(dA);
		thrust::device_ptr<double> A_end(dA + ARRAY_SIZE);
		thrust::sequence(thrust::cuda::par, A_begin, A_end);
	}
	cudaMemcpy(hA, dA, sizeof(double) * ARRAY_SIZE, cudaMemcpyDeviceToHost);

	bool correct = true;
	for (int i = 0; i < ARRAY_SIZE; i++)
		if (hA[i] != 1.0 * i) {
			correct = false;
			break;
		}

	if (correct)
		cout << "Old sequence: correct" << endl;
	else
		cout << "Old sequence: INCORRECT" << endl;

	free(hA);
	cudaFree(dA);
}

void new_sequence() {
	const int ARRAY_SIZE = 1000;

	double *mA;

	cudaMallocManaged(&mA, sizeof(double) * ARRAY_SIZE);

	for (int i = 0; i < ARRAY_SIZE; i++)
		mA[i] = 0.0;

	/* FIXME: it's not correct to call
	    thrust::sequence(thrust::cuda::par, mA, mA + ARRAY_SIZE);
	   */
	{
		thrust::device_ptr<double> A_begin(mA);
		thrust::device_ptr<double> A_end(mA + ARRAY_SIZE);
		thrust::sequence(thrust::cuda::par, A_begin, A_end);
	}
	cudaDeviceSynchronize();

	bool correct = true;
	for (int i = 0; i < ARRAY_SIZE; i++)
		if (mA[i] != 1.0 * i) {
			correct = false;
			break;
		}

	if (correct)
		cout << "New sequence: correct" << endl;
	else
		cout << "New sequence: INCORRECT" << endl;

	cudaFree(mA);
}

int main(int argc, char **argv) 
{
	old_fill();
	new_fill();
	old_copy();
	new_copy();
	old_sequence();
	new_sequence();
	return 0;
}
