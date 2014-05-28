#ifndef MC64_CUH
#define MC64_CUH

namespace mc64{

namespace device{

__global__ void
getResidualValues(int        N,
		          double*    c_vals,
				  double*    max_vals,
				  const int* row_ptr)
{
	int bid = blockIdx.x + blockIdx.y * gridDim.x;

	if (bid >= N)
		return;

	__shared__ int    start;
	__shared__ int    end;
	__shared__ double max_val;

	if (threadIdx.x == 0) {
		start = row_ptr[bid];
		end = row_ptr[bid + 1];
		max_val = max_vals[bid];
	}
	__syncthreads();

	for (int i = threadIdx.x + start; i < end; i += blockDim.x)
		c_vals[i] = log(max_val / c_vals[i]);
}

__global__ void
updateMatrix(int           N,
		     const int*    row_offsets,
		     const int*    row_perm,
			 const double* row_scale,
			 const double* col_scale,
			 int*          row_indices,
			 const int*    column_indices,
			 double*       values)
{
	int idx = blockIdx.x + blockIdx.y * gridDim.x;
	if (idx >= N) return;
	int start_idx = row_offsets[idx], end_idx = row_offsets[idx + 1];

	int new_row = row_perm[idx];
	double row_fact = row_scale[idx];
	for (int tid = threadIdx.x + start_idx; tid < end_idx; tid += blockDim.x) {
		row_indices[tid] = new_row;
		values[tid] *= (row_fact * col_scale[column_indices[tid]]);
	}
}

__global__ void
updateMatrix(int           N,
		     const int*    row_offsets,
		     const int*    row_perm,
			 int*          row_indices)
{
	int idx = blockIdx.x + blockIdx.y * gridDim.x;
	if (idx >= N) return;
	int start_idx = row_offsets[idx], end_idx = row_offsets[idx + 1];

	int new_row = row_perm[idx];
	for (int tid = threadIdx.x + start_idx; tid < end_idx; tid += blockDim.x)
		row_indices[tid] = new_row;
}

} // end namespace device
} // end namespace mc64

#endif
