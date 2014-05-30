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

void old_transform() {
	const int ARRAY_SIZE = 1000;

	double *hA, *dA;
	hA = (double *) malloc(ARRAY_SIZE * sizeof(double));
	cudaMalloc(&dA, ARRAY_SIZE * sizeof(double));

	for (int i = 0; i < ARRAY_SIZE; i++)
		hA[i] = 1.0 * (i + 1);

	cudaMemcpy(dA, hA, sizeof(double) * ARRAY_SIZE, cudaMemcpyHostToDevice);
	/* FIXME: it is not correct to use
	thrust::transform(thrust::cuda::par, dA, dA + ARRAY_SIZE, dA, thrust::negate<double>());
	*/
	{
		thrust::device_ptr<double> A_begin(dA);
		thrust::device_ptr<double> A_end(dA + ARRAY_SIZE);
		thrust::transform(thrust::cuda::par, A_begin, A_end, A_begin, thrust::negate<double>());
	}
	cudaMemcpy(hA, dA, sizeof(double) * ARRAY_SIZE, cudaMemcpyDeviceToHost);

	bool correct = true;
	for (int i = 0; i < ARRAY_SIZE; i++)
		if (hA[i] != - 1.0 * (i + 1)) {
			correct = false;
			break;
		}

	if (correct)
		cout << "Old transform: correct" << endl;
	else
		cout << "Old transform: INCORRECT" << endl;

	cudaFree(dA);
	free(hA);
}

void new_transform() {
	const int ARRAY_SIZE = 1000;

	double *mA;
	cudaMallocManaged(&mA, ARRAY_SIZE * sizeof(double));

	for (int i = 0; i < ARRAY_SIZE; i++)
		mA[i] = 1.0 * (i + 1);

	/* FIXME: it is not correct to use
	thrust::transform(thrust::cuda::par, mA, mA + ARRAY_SIZE, mA, thrust::negate<double>());
	*/
	{
		thrust::device_ptr<double> A_begin(mA);
		thrust::device_ptr<double> A_end(mA + ARRAY_SIZE);
		thrust::transform(thrust::cuda::par, A_begin, A_end, A_begin, thrust::negate<double>());
	}
	cudaDeviceSynchronize();

	bool correct = true;
	for (int i = 0; i < ARRAY_SIZE; i++)
		if (mA[i] != - 1.0 * (i + 1)) {
			correct = false;
			break;
		}

	if (correct)
		cout << "New transform: correct" << endl;
	else
		cout << "New transform: INCORRECT" << endl;

	cudaFree(mA);
}

void old_transform_if() {
	const int ARRAY_SIZE = 1000;

	double *hA, *dA, *hB, *dB;
	int *h_stencil, *d_stencil;

	hA = (double *) malloc(ARRAY_SIZE * sizeof(double));
	cudaMalloc(&dA, ARRAY_SIZE * sizeof(double));
	hB = (double *) malloc(ARRAY_SIZE * sizeof(double));
	cudaMalloc(&dB, ARRAY_SIZE * sizeof(double));
	h_stencil = (int *) malloc(ARRAY_SIZE * sizeof(int));
	cudaMalloc(&d_stencil, ARRAY_SIZE * sizeof(int));

	for (int i = 0; i < ARRAY_SIZE; i++)
		hB[i] = hA[i] = 1.0 * (i + 1);

	for (int i = 0; i < ARRAY_SIZE; i++) {
		if (i < (ARRAY_SIZE >> 1))
			h_stencil[i] = 1;
		else
			h_stencil[i] = 0;
	}

	cudaMemcpy(dA, hA, sizeof(double) * ARRAY_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, hB, sizeof(double) * ARRAY_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(d_stencil, h_stencil, sizeof(int) * ARRAY_SIZE, cudaMemcpyHostToDevice);
	/* FIXME: it is not correct to use
	thrust::transform_if(thrust::cuda::par, dA, dA + ARRAY_SIZE, dB, d_stencil, dA, thrust::plus<double>(), thrust::identity<int>());
	*/
	{
		thrust::device_ptr<double> A_begin(dA);
		thrust::device_ptr<double> A_end(dA + ARRAY_SIZE);
		thrust::device_ptr<double> B_begin(dB);
		thrust::device_ptr<int>    stencil_begin(d_stencil);
		thrust::transform_if(thrust::cuda::par, A_begin, A_end, B_begin, stencil_begin, A_begin, thrust::plus<double>(), thrust::identity<int>());
	}
	cudaMemcpy(hA, dA, sizeof(double) * ARRAY_SIZE, cudaMemcpyDeviceToHost);

	bool correct = true;
	for (int i = 0; i < (ARRAY_SIZE >> 1); i++)
		if (hA[i] != 2.0 * (i + 1)) {
			correct = false;
			break;
		}

	if (correct) {
		for (int i = (ARRAY_SIZE >> 1); i < ARRAY_SIZE; i++)
			if (hA[i] != 1.0 * (i + 1)) {
				correct = false;
				break;
			}
	}

	if (correct)
		cout << "Old transform_if: correct" << endl;
	else
		cout << "Old transform_if: INCORRECT" << endl;

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(d_stencil);
	free(hA);
	free(hB);
	free(h_stencil);
}

void new_transform_if() {
	const int ARRAY_SIZE = 1000;

	double *mA, *mB;
	int *m_stencil;

	cudaMallocManaged(&mA, ARRAY_SIZE * sizeof(double));
	cudaMallocManaged(&mB, ARRAY_SIZE * sizeof(double));
	cudaMallocManaged(&m_stencil, ARRAY_SIZE * sizeof(int));

	for (int i = 0; i < ARRAY_SIZE; i++)
		mB[i] = mA[i] = 1.0 * (i + 1);

	for (int i = 0; i < ARRAY_SIZE; i++) {
		if (i < (ARRAY_SIZE >> 1))
			m_stencil[i] = 1;
		else
			m_stencil[i] = 0;
	}

	/* FIXME: it is not correct to use 
	thrust::transform_if(thrust::cuda::par, mA, mA + ARRAY_SIZE, mB, m_stencil, mA, thrust::plus<double>(), thrust::identity<int>());
	*/

	{ 
		thrust::device_ptr<double> A_begin(mA);
		thrust::device_ptr<double> A_end(mA + ARRAY_SIZE);
		thrust::device_ptr<double> B_begin(mB);
		thrust::device_ptr<int>    stencil_begin(m_stencil);
		thrust::transform_if(thrust::cuda::par, A_begin, A_end, B_begin, stencil_begin, A_begin, thrust::plus<double>(), thrust::identity<int>());
	}
	cudaDeviceSynchronize();

	bool correct = true;
	for (int i = 0; i < (ARRAY_SIZE >> 1); i++)
		if (mA[i] != 2.0 * (i + 1)) {
			correct = false;
			break;
		}

	if (correct) {
		for (int i = (ARRAY_SIZE >> 1); i < ARRAY_SIZE; i++)
			if (mA[i] != 1.0 * (i + 1)) {
				correct = false;
				break;
			}
	}

	if (correct)
		cout << "New transform_if: correct" << endl;
	else
		cout << "New transform_if: INCORRECT" << endl;

	cudaFree(mA);
	cudaFree(mB);
	cudaFree(m_stencil);
}

int main(int argc, char **argv) 
{
	old_transform();
	new_transform();
	old_transform_if();
	new_transform_if();
	return 0;
}
