#include <algorithm>
#include <fstream>
#include <cmath>
#include <string>

extern "C" {
#include "mm_io/mm_io.h"
}
#include "mc64/mc64.h"

typedef typename thrust::host_vector<int>     IntVectorH;
typedef typename thrust::host_vector<double>  DoubleVectorH;

// -----------------------------------------------------------------------------

enum TestColor {
  COLOR_NO = 0,
  COLOR_RED,
  COLOR_GREEN
};

// -----------------------------------------------------------------------------

class OutputItem
{
public:
  OutputItem(std::ostream &o) : m_o(o), m_additional_item_count(19) {}

  int  m_additional_item_count;

  template <typename T>
  void operator() (T item, TestColor c = COLOR_NO)
  {
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

// -----------------------------------------------------------------------------

int main(int argc, char **argv)
{
  if (argc < 2)
    return 1;

  // Read matrix from file (COO format)
  MM_typecode matcode;
  int M, N, nnz;
  int* row_i;
  int* col_j;
  double* vals;

  if (mm_read_mtx_crd(argv[1], &M, &N, &nnz, &row_i, &col_j, &vals, &matcode) != 0)
    return 1;

  // Switch to 0-based indices
  for (int i = 0; i < nnz; i++) {
    row_i[i]--;
    col_j[i]--;
  }

  // Convert to CSR format and load into thrust vectors.
  IntVectorH    row_offsets(N + 1);
  IntVectorH    column_indices(nnz);
  DoubleVectorH values(nnz);

  mc64::coo2csr(M, N, nnz, row_i, col_j, vals, row_offsets, column_indices, values);

  // Run the MC64 algorithm and generate output
  mc64::MC64 algo(row_offsets, column_indices, values);

  OutputItem outputItem(std::cout);
  std::string matname;

  mc64::getName(argv[1], matname);

  std::cout << "<tr valign=top>" << std::endl;

  outputItem(matname);
  outputItem(N);
  outputItem(nnz);

  try {
    algo.execute();
  } catch (const mc64::system_error& se) {
    outputItem("");
    outputItem("");
    outputItem("");
    outputItem("");
    outputItem("");
    return 1;
  }

  outputItem(algo.getTimePre());
  outputItem(algo.getTimeFirst());
  outputItem(algo.getTimeSecond());
  outputItem(algo.getTimePost());
  outputItem(algo.getTimeTotal());

  std::cout << "</tr>" << std::endl;

  return 0;
}
