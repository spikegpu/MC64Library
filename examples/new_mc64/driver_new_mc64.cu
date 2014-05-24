#include <algorithm>
#include <fstream>
#include <cmath>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <mc64/mc64.h>
#include <mc64/timer.h>

using std::cout;
using std::cin;
using std::endl;
using std::string;
using std::vector;

typedef typename thrust::host_vector<int>     IntVectorH;
typedef typename thrust::host_vector<double>  DoubleVectorH;

typedef typename mc64::ManagedVector<int>     IntVector;
typedef typename mc64::ManagedVector<double>  DoubleVector;

int main(int argc, char **argv)
{
	size_t        N, nnz;

	IntVector     row_offsets_managed;
	IntVector     column_indices_managed;
	DoubleVector  values_managed;

	std::ifstream  fin;

	if (argc < 2)
		return 1;
	else {
		fin.open(argv[1], std::ifstream::in);
		if (!fin.is_open())
			return 1;
	}

	fin >> N >> nnz;

	row_offsets_managed.resize(N + 1);
	column_indices_managed.resize(nnz);
	values_managed.resize(nnz);

	for (int i = 0; i <= N; i++)
		fin >> row_offsets_managed[i];

	for (int i = 0; i < nnz; i++)
		fin >> column_indices_managed[i];

	for (int i = 0; i < nnz; i++)
		fin >> values_managed[i];

	fin.close();

	mc64::NewMC64 newMC64(row_offsets_managed, column_indices_managed, values_managed);

	newMC64.execute();

	cout << newMC64.getTimeTotal() << endl;
	cout << newMC64.getTimePre() << endl;
	cout << newMC64.getTimeFirst() << endl;
	cout << newMC64.getTimeSecond() << endl;
	cout << newMC64.getTimePost() << endl;

	return 0;
}
