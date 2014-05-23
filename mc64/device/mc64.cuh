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

} // end namespace device
} // end namespace mc64

#endif
