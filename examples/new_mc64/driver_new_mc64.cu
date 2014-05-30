#include <algorithm>
#include <fstream>
#include <cmath>
#include <string>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>

#include <mc64/mc64.h>
#include <mc64/timer.h>

using std::cout;
using std::cerr;
using std::cin;
using std::endl;
using std::string;
using std::vector;

typedef typename thrust::host_vector<int>     IntVectorH;
typedef typename thrust::host_vector<double>  DoubleVectorH;

typedef typename mc64::ManagedVector<int>     IntVector;
typedef typename mc64::ManagedVector<double>  DoubleVector;

// Color to print
enum TestColor {COLOR_NO = 0,
                COLOR_RED,
                COLOR_GREEN} ;


class OutputItem
{
public:
	OutputItem(std::ostream &o): m_o(o), m_additional_item_count(19) {}

	int           m_additional_item_count;

	template <typename T>
	void operator() (T item, TestColor c = COLOR_NO) {
		m_o << "<td style=\"border-style: inset;\">\n";
		switch (c)
		{
			case COLOR_RED:
				m_o << "<p> <FONT COLOR=\"Red\">" << item << " </FONT> </p>\n";
				break;

			case COLOR_GREEN:
				m_o << "<p> <FONT COLOR=\"Green\">" << item << " </FONT> </p>\n";
				break;

			default:
				m_o << "<p> " << item << " </p>\n";
				break;
		}
		m_o << "</td>\n";
	}
private:
	std::ostream &m_o;
};

int main(int argc, char **argv)
{
	size_t        N, nnz;
	std::string   fileMat;

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

	fin >> fileMat >> N >> nnz;

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

	OutputItem outputItem(cout);

	cout << "<tr valign=top>" << endl;

	outputItem(fileMat);

	outputItem(N);

	outputItem(nnz);

	mc64::NewMC64 newMC64(row_offsets_managed, column_indices_managed, values_managed);

	try {
		newMC64.execute();
	} catch (const mc64::system_error& se) {
		outputItem("");
		outputItem("");
		outputItem("");
		outputItem("");
		outputItem("");

		return 1;
	}

	outputItem(newMC64.getTimePre());

	outputItem(newMC64.getTimeFirst());

	outputItem(newMC64.getTimeSecond());

	outputItem(newMC64.getTimePost());

	outputItem(newMC64.getTimeTotal());

	cout << "</tr>" << endl;

	return 0;
}
