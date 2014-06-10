#include <algorithm>
#include <fstream>
#include <cmath>
#include <string>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "mc64/mc64.h"

using std::cout;
using std::cin;
using std::endl;
using std::string;
using std::vector;

typedef typename thrust::host_vector<int>     IntVectorH;
typedef typename thrust::host_vector<double>  DoubleVectorH;

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

	IntVectorH    row_offsets;
	IntVectorH    column_indices;
	DoubleVectorH values;

	std::ifstream  fin;

	std::string fileMat;

	if (argc < 2)
		return 1;
	else {
		fin.open(argv[1], std::ifstream::in);
		if (!fin.is_open())
			return 1;
	}

	fin >> fileMat >> N >> nnz;

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

	OutputItem outputItem(cout);

	cout << "<tr valign=top>" << endl;

	outputItem(fileMat);

	outputItem(N);

	outputItem(nnz);

	try {
		oldMC64.execute();
	} catch (const mc64::system_error& se) {
		outputItem("");
		outputItem("");
		outputItem("");
		outputItem("");
		outputItem("");

		return 1;
	}

	outputItem(oldMC64.getTimePre());

	outputItem(oldMC64.getTimeFirst());

	outputItem(oldMC64.getTimeSecond());

	outputItem(oldMC64.getTimePost());

	outputItem(oldMC64.getTimeTotal());

	cout << "</tr>" << endl;

	return 0;
}
