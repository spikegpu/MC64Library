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

void old_sort(bool decrease = false) {
	const int ARRAY_SIZE = 1000;

	double *hA, *dA;

	hA = (double *) malloc (sizeof(double) * ARRAY_SIZE);
	cudaMalloc(&dA, sizeof(double) * ARRAY_SIZE);

	for (int i = 0; i < ARRAY_SIZE; i++)
		hA[i] = 1.0 * (rand() % ARRAY_SIZE);

	cudaMemcpy(dA, hA, sizeof(double) * ARRAY_SIZE, cudaMemcpyHostToDevice);

	if (decrease) {
		thrust::device_ptr<double> dA_begin = thrust::device_pointer_cast(dA);
		thrust::device_ptr<double> dA_end   = thrust::device_pointer_cast(dA + ARRAY_SIZE);
		thrust::sort(thrust::cuda::par, dA_begin, dA_end, thrust::greater<double>());

		/* FIXME: The program gets seg-fault if we do:
		   thrust::sort(thrust::cuda::par, dA, dA + ARRAY_SIZE, thrust::greater<double>());
		   which does not seem to make much sense.
		   */
	}
	else
		thrust::sort(thrust::cuda::par, dA, dA + ARRAY_SIZE);

	cudaMemcpy(hA, dA, sizeof(double) * ARRAY_SIZE, cudaMemcpyDeviceToHost);

	bool correct = true;
	for (int i = 1; i < ARRAY_SIZE; i++)
		if (decrease ? (hA[i] > hA[i-1]) : (hA[i] < hA[i-1])) {
			correct = false;
			break;
		}

	if (correct)
		cout << "Old sort correct" << endl;
	else
		cout << "Old sort INCORRECT" << endl;

	cudaFree(dA);
	free(hA);
}

void new_sort(bool decrease = false) {
	const int ARRAY_SIZE = 1000;

	double *mA;

	cudaMallocManaged(&mA, sizeof(double) * ARRAY_SIZE);

	for (int i = 0; i < ARRAY_SIZE; i++)
		mA[i] = 1.0 * (rand() % ARRAY_SIZE);

	if (decrease)
		thrust::sort(thrust::cuda::par, mA, mA + ARRAY_SIZE, thrust::greater<double>());
	else
		thrust::sort(thrust::cuda::par, mA, mA + ARRAY_SIZE);

	cudaDeviceSynchronize();

	bool correct = true;
	for (int i = 1; i < ARRAY_SIZE; i++) {
		if (decrease ? (mA[i] > mA[i-1]) : (mA[i] < mA[i-1])) {
			correct = false;
			break;
		}
	}

	if (correct)
		cout << "New sort correct" << endl;
	else
		cout << "New sort INCORRECT" << endl;

	cudaFree(mA);
}

void old_sort_by_key(bool decrease = false) {
	const int ARRAY_SIZE = 10;

	int    *h_keys,   *d_keys;
	double *h_values, *d_values;

	h_keys = (int *)malloc(sizeof(int) * ARRAY_SIZE);
	h_values = (double *)malloc(sizeof(double) * ARRAY_SIZE);

	cudaMalloc(&d_keys, sizeof(int) * ARRAY_SIZE);
	cudaMalloc(&d_values, sizeof(double) * ARRAY_SIZE);

	cout << "Old before: " << endl;
	for (int i = 0; i < ARRAY_SIZE; i++) {
		h_keys[i] = rand() % (ARRAY_SIZE >> 1);
		h_values[i] = 1.0 * (rand() % ARRAY_SIZE);
		cout << "(" << h_keys[i] << ", " << h_values[i] << ")" << endl;
	}

	cudaMemcpy(d_keys, h_keys, sizeof(int) * ARRAY_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(d_values, h_values, sizeof(double) * ARRAY_SIZE, cudaMemcpyHostToDevice);

	if (decrease) {
		thrust::device_ptr<int> keys_begin = thrust::device_pointer_cast(d_keys);
		thrust::device_ptr<int> keys_end   = thrust::device_pointer_cast(d_keys + ARRAY_SIZE);
		thrust::device_ptr<double> values_begin = thrust::device_pointer_cast(d_values);
		thrust::sort_by_key(thrust::cuda::par, keys_begin, keys_end, values_begin, thrust::greater<int>());
		/* FIXME: The program gets wrong results if we do:
		   thrust::sort_by_key(thrust::cuda::par, d_keys, d_keys + ARRAY_SIZE, d_values, thrust::greater<int>());
		   which does not seem to make much sense.
		   */
	}
	else
		thrust::sort_by_key(thrust::cuda::par, d_keys, d_keys + ARRAY_SIZE, d_values);

	cudaMemcpy(h_keys, d_keys, sizeof(int) * ARRAY_SIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_values, d_values, sizeof(double) * ARRAY_SIZE, cudaMemcpyDeviceToHost);

	cout << "Old after: " << endl;
	for (int i = 0; i < ARRAY_SIZE; i++) {
		cout << "(" << h_keys[i] << ", " << h_values[i] << ")" << endl;
	}


	cudaFree(d_keys);
	cudaFree(d_values);
	free(h_keys);
	free(h_values);
}

void new_sort_by_key(bool decrease = false) {
	const int ARRAY_SIZE = 10;

	int    *m_keys;
	double *m_values;

	cudaMallocManaged(&m_keys, sizeof(int) * ARRAY_SIZE);
	cudaMallocManaged(&m_values, sizeof(double) * ARRAY_SIZE);

	cout << "New before: " << endl;
	for (int i = 0; i < ARRAY_SIZE; i++) {
		m_keys[i] = rand() % (ARRAY_SIZE >> 1);
		m_values[i] = 1.0 * (rand() % ARRAY_SIZE);
		cout << "(" << m_keys[i] << ", " << m_values[i] << ")" << endl;
	}

	if (decrease) {
		thrust::device_ptr<int> keys_begin = thrust::device_pointer_cast(m_keys);
		thrust::device_ptr<int> keys_end   = thrust::device_pointer_cast(m_keys + ARRAY_SIZE);
		thrust::device_ptr<double> values_begin = thrust::device_pointer_cast(m_values);
		thrust::sort_by_key(thrust::cuda::par, keys_begin, keys_end, values_begin, thrust::greater<int>());
		/* FIXME: The program gets wrong results if we do:
		   thrust::sort_by_key(thrust::cuda::par, d_keys, d_keys + ARRAY_SIZE, d_values, thrust::greater<int>());
		   which does not seem to make much sense.
		   Also note that the behavior of sort_by_keys seems to be different from that of sort.
		   */
	}
	else
		thrust::sort_by_key(thrust::cuda::par, m_keys, m_keys + ARRAY_SIZE, m_values);
	cudaDeviceSynchronize();

	cout << "New after: " << endl;
	for (int i = 0; i < ARRAY_SIZE; i++)
		cout << "(" << m_keys[i] << ", " << m_values[i] << ")" << endl;


	cudaFree(m_keys);
	cudaFree(m_values);
}

int main(int argc, char **argv)
{
	old_sort(false);
	new_sort(false);
	old_sort(true);
	new_sort(true);
	old_sort_by_key(false);
	old_sort_by_key(true);
	new_sort_by_key(false);
	new_sort_by_key(true);
	return 0;
}
