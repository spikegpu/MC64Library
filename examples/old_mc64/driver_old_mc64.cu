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

	IntVectorH    row_offsets;
	IntVectorH    column_indices;
	DoubleVectorH values;

	std::ifstream  fin;

	if (argc < 2)
		return 1;
	else {
		fin.open(argv[1], std::ifstream::in);
		if (!fin.is_open())
			return 1;
	}

	fin >> N >> nnz;

	row_offsets.resize(N + 1);
	column_indices.resize(nnz);
	values.resize(nnz);

	for (int i = 0; i <= N; i++)
		fin >> row_offsets[i];

	for (int i = 0; i < nnz; i++)
		fin >> column_indices[i];

	for (int i = 0; i < nnz; i++)
		fin >> values[i];

	fin.close();

	mc64::OldMC64 oldMC64(row_offsets, column_indices, values);

	oldMC64.execute();
	cout << oldMC64.getTimeTotal() << endl;
	cout << oldMC64.getTimePre() << endl;
	cout << oldMC64.getTimeFirst() << endl;
	cout << oldMC64.getTimeSecond() << endl;
	cout << oldMC64.getTimePost() << endl;

	return 0;
}
