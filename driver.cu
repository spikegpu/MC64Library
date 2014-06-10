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

int main(int argc, char **argv)
{
  if (argc < 3) {
    std::cout << "Usage:\n  driver input_file output_file" << std::endl;
    return 1;
  }

  // Read matrix from file (COO format)
  MM_typecode matcode;
  int M, N, nnz;
  int* row_i;
  int* col_j;
  double* vals;

  std::cout << "Read MTX file... ";
  int err = mm_read_mtx_crd(argv[1], &M, &N, &nnz, &row_i, &col_j, &vals, &matcode);
  if (err != 0) {
    std::cout << "error: " << err << std::endl;
    return 1;
  }
  std::cout << "M = " << M << " N = " << N << " nnz = " << nnz << std::endl;

  // Switch to 0-based indices
  for (int i = 0; i < nnz; i++) {
    row_i[i]--;
    col_j[i]--;
  }

  // Convert to CSR format and load into thrust vectors.
  IntVectorH    row_offsets(N + 1);
  IntVectorH    column_indices(nnz);
  DoubleVectorH values(nnz);

  std::cout << "Convert to CSR" << std::endl;
  mc64::coo2csr(M, N, nnz, row_i, col_j, vals, row_offsets, column_indices, values);

  // Print thrust vectors
  /*
  std::cout << "Row offsets\n";
  thrust::copy(row_offsets.begin(), row_offsets.end(), std::ostream_iterator<int>(std::cout, "\n"));
  std::cout << "Column indices\n";
  thrust::copy(column_indices.begin(), column_indices.end(), std::ostream_iterator<int>(std::cout, "\n"));
  std::cout << "Values\n";
  thrust::copy(values.begin(), values.end(), std::ostream_iterator<double>(std::cout, "\n"));
  */

  // Run the MC64 algorithm
  mc64::OldMC64 algo(row_offsets, column_indices, values);

  std::cout << "Run MC64... ";
  try {
    algo.execute();
  } catch (const mc64::system_error& se) {
    std::cout << "error " << se.reason() << std::endl;
    return 1;
  }
  std::cout << "success" << std::endl;

  // Generate output with results.
  std::cout << "Time pre:    " << algo.getTimePre()    << std::endl;
  std::cout << "Time first:  " << algo.getTimeFirst()  << std::endl;
  std::cout << "Time second: " << algo.getTimeSecond() << std::endl;
  std::cout << "Time pos:    " << algo.getTimePost()   << std::endl;
  std::cout << "Time total:  " << algo.getTimeTotal()  << std::endl;

  std::cout << "Write output file " << argv[2] << std::endl;
  std::ofstream fout;
  fout.open(argv[2]);
  algo.print(fout);
  fout.close();

  return 0;
}
