#ifndef MC64_UM_H
#define MC64_UM_H

#include "mc64/mc64.h"

namespace mc64  {

class NewMC64: public MC64
{
private:
	typedef ManagedVector<int>                      IntVector;
	typedef ManagedVector<double>                   DoubleVector;
	typedef ManagedVector<bool>                     BoolVector;
	typedef thrust::host_vector<int>                IntVectorH;
	typedef thrust::host_vector<bool>               BoolVectorH;
	typedef thrust::host_vector<double>             DoubleVectorH;

	IntVector      m_row_offsets;
	IntVector      m_column_indices;
	DoubleVector   m_values;

	IntVector      m_rowPerm;
	DoubleVector   m_rowScale;
	DoubleVector   m_colScale;

	void formBipartiteGraph(DoubleVector &c_val, DoubleVector &max_val_in_col);

	void initPartialMatch(DoubleVector& c_val,
						  DoubleVector& rowScale,
						  DoubleVector& colScale,
						  IntVector&    rowReordering,
						  IntVector&    rev_match_nodes,
						  BoolVector&   matched,
						  BoolVector&   rev_matched);

	void findShortestAugPath(int            init_node,
							 BoolVector&    matched,
							 BoolVector&    rev_matched,
							 IntVector&     match_nodes,
							 IntVector&     rev_match_nodes,
							 IntVector&     prev,
							 DoubleVector&  u_val,
							 DoubleVector&  v_val,
							 DoubleVector&  c_val,
							 IntVector&     irn);

public:
	NewMC64(const IntVector&    row_offsets,
			const IntVector&    column_indices,
			const DoubleVector& values)
	:       m_row_offsets(row_offsets),
	        m_column_indices(column_indices),
	        m_values(values)
		    {
				size_t n = row_offsets.size() - 1;
				m_rowPerm.resize(n);
				m_rowScale.resize(n);
				m_colScale.resize(n);
				m_n   = n;
				m_nnz = m_values.size();
			}

	virtual ~NewMC64() {}

	void execute(bool scale = true);
	void print(std::ostream &o);
};

void
NewMC64::execute(bool scale)
{
	DoubleVector c_val(m_nnz);
	DoubleVector max_val_in_col(m_n, 0);

	CPUTimer total_timer;
	CPUTimer cpu_timer;

	total_timer.Start();
	cpu_timer.Start();
	formBipartiteGraph(c_val, max_val_in_col);
	cpu_timer.Stop();
	m_time_pre = cpu_timer.getElapsed();

	cpu_timer.Start();
	DoubleVector&  rowScale = m_rowScale;
	DoubleVector&  colScale = m_colScale;
	IntVector      rowReordering(m_n);
	IntVector      rev_match_nodes(m_nnz);
	BoolVector     matched(m_n, false);
	BoolVector     rev_matched(m_n, false);

	initPartialMatch(c_val, colScale, rowScale, rowReordering, rev_match_nodes, matched, rev_matched);
	cpu_timer.Stop();
	m_time_first = cpu_timer.getElapsed();

	cpu_timer.Start();
	{
		IntVector    irn(m_n);
		IntVector    prev(m_n);
		for (int i = 0; i < m_n; i++) {
			if(rev_matched[i]) continue;
			findShortestAugPath(i, matched, rev_matched, rowReordering, rev_match_nodes,  prev, colScale, rowScale, c_val, irn);
		}

		if (thrust::any_of(matched.begin(), matched.end(), thrust::logical_not<bool>()))
			throw system_error(system_error::Matrix_singular, "Singular matrix found");

		thrust::transform(colScale.begin(), colScale.end(), colScale.begin(), Exponential());
		thrust::transform(thrust::make_transform_iterator(rowScale.begin(), Exponential()),
				thrust::make_transform_iterator(rowScale.end(), Exponential()),
				max_val_in_col.begin(),
				rowScale.begin(),
				thrust::divides<double>());

		cudaDeviceSynchronize();
	}
	cpu_timer.Stop();
	m_time_second = cpu_timer.getElapsed();

	cpu_timer.Start();
	thrust::scatter(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(m_n), rowReordering.begin(), m_rowPerm.begin());


	IntVector&    rowPerm  = m_rowPerm;
	IntVector     row_indices(m_nnz);

	if (scale) {
		const int *p_row_offsets  = thrust::raw_pointer_cast(&m_row_offsets[0]);
		const int *p_row_perm     = thrust::raw_pointer_cast(&rowPerm[0]);
		const double *p_row_scale = thrust::raw_pointer_cast(&rowScale[0]);
		const double *p_col_scale = thrust::raw_pointer_cast(&colScale[0]);
		int*  p_row_indices       = thrust::raw_pointer_cast(&row_indices[0]);
		const int *p_column_indices = thrust::raw_pointer_cast(&m_column_indices[0]);
		double* p_values          = thrust::raw_pointer_cast(&m_values[0]);
		int gridX = m_n, gridY = 1;
		kernelConfigAdjust(gridX, gridY, MAX_GRID_DIMENSION);
		dim3 grids(gridX, gridY);
		device::updateMatrix<<<grids, 64>>>(m_n, p_row_offsets, p_row_perm, p_row_scale, p_col_scale, p_row_indices, p_column_indices, p_values);
	} else {
		const int *p_row_offsets  = thrust::raw_pointer_cast(&m_row_offsets[0]);
		const int *p_row_perm     = thrust::raw_pointer_cast(&rowPerm[0]);
		int*  p_row_indices       = thrust::raw_pointer_cast(&row_indices[0]);
		int gridX = m_n, gridY = 1;
		kernelConfigAdjust(gridX, gridY, MAX_GRID_DIMENSION);
		dim3 grids(gridX, gridY);
		device::updateMatrix<<<grids, 64>>>(m_n, p_row_offsets, p_row_perm, p_row_indices);
	}

	{
		thrust::stable_sort_by_key(row_indices.begin(), row_indices.end(), thrust::make_zip_iterator(thrust::make_tuple(m_column_indices.begin(), m_values.begin())));
		indices_to_offsets(row_indices, m_row_offsets);
	}
	cudaDeviceSynchronize();
	cpu_timer.Stop();
	m_time_post = cpu_timer.getElapsed();

	m_done = true;

	total_timer.Stop();
	m_time_total = total_timer.getElapsed();
}

void
NewMC64::formBipartiteGraph(DoubleVector &c_val, DoubleVector &max_val_in_col)
{
	IntVector&    row_offsets  = m_row_offsets;

	{
		IntVector    row_indices(m_nnz);
		offsets_to_indices(row_offsets, row_indices);
		thrust::transform(m_values.begin(), m_values.end(), c_val.begin(), AbsoluteValue<double>());
		thrust::reduce_by_key(row_indices.begin(), row_indices.end(), c_val.begin(), thrust::make_discard_iterator(), max_val_in_col.begin(), thrust::equal_to<double>(), thrust::maximum<double>());
	}

	double *p_c_val          = thrust::raw_pointer_cast(&c_val[0]);
	const int *p_row_offsets = thrust::raw_pointer_cast(&row_offsets[0]);
	double *p_max_val        = thrust::raw_pointer_cast(&max_val_in_col[0]);

	int blockX = m_n, blockY = 1;
	kernelConfigAdjust(blockX, blockY, MAX_GRID_DIMENSION);
	dim3 grids(blockX, blockY);

	device::getResidualValues<<<grids, 64>>>(m_n, p_c_val, p_max_val, p_row_offsets); 
	cudaDeviceSynchronize();
}

void
NewMC64::initPartialMatch(DoubleVector& c_val,
						  DoubleVector& rowScale,
						  DoubleVector& colScale,
						  IntVector&    rowReordering,
						  IntVector&    rev_match_nodes,
						  BoolVector&   matched,
						  BoolVector&   rev_matched)
{
	{
		IntVector row_indices(c_val.size());
		offsets_to_indices(m_row_offsets, row_indices);
		thrust::reduce_by_key(row_indices.begin(), row_indices.end(), c_val.begin(), thrust::make_discard_iterator(), rowScale.begin(), thrust::equal_to<double>(), thrust::minimum<double>());
	}

	thrust::fill(colScale.begin(), colScale.end(), LOC_INFINITY);
	cudaDeviceSynchronize();


	for(int i = 0; i < m_n; i++) {
		int start_idx = m_row_offsets[i], end_idx = m_row_offsets[i+1];
		int min_idx = -1;
		for(int j = start_idx; j < end_idx; j++) {
			if (c_val[j] > LOC_INFINITY / 2.0) continue;
			int row = m_column_indices[j];
			double tmp_val = c_val[j] - rowScale[row];
			if(colScale[i] > tmp_val) {
				colScale[i] = tmp_val;
				min_idx = j;
			}
		}
		if(min_idx >= 0) {
			int tmp_row = m_column_indices[min_idx];
			if(!matched[tmp_row]) {
				rev_matched[i] = true;
				matched[tmp_row] = true;
				rowReordering[tmp_row] = i;
				rev_match_nodes[i] = min_idx;
			}
		}
	}

	thrust::transform_if(rowScale.begin(), rowScale.end(), matched.begin(), rowScale.begin(), ClearValue(), thrust::logical_not<bool>());
	thrust::transform_if(colScale.begin(), colScale.end(), rev_matched.begin(), colScale.begin(), ClearValue(), thrust::logical_not<bool>());
	cudaDeviceSynchronize();
}

void
NewMC64::findShortestAugPath(int            init_node,
							 BoolVector&    matched,
							 BoolVector&    rev_matched,
							 IntVector&     match_nodes,
							 IntVector&     rev_match_nodes,
							 IntVector&     prev,
							 DoubleVector&  u_val,
							 DoubleVector&  v_val,
							 DoubleVector&  c_val,
							 IntVector&     irn)
{
	static IntVector B(m_n, 0);
	int b_cnt = 0;
	static BoolVector inB(m_n, false);

	std::priority_queue<Dijkstra, std::vector<Dijkstra>, CompareValue<double> > Q;

	double lsp = 0.0;
	double lsap = LOC_INFINITY;
	int cur_node = init_node;

	int i;

	int isap = -1;
	int ksap = -1;
	prev[init_node] = -1;

	static DoubleVector d_vals(m_n, LOC_INFINITY);
	static BoolVector visited(m_n, false);

	while(1) {
		int start_cur = m_row_offsets[cur_node];
		int end_cur = m_row_offsets[cur_node+1];
		for(i = start_cur; i < end_cur; i++) {
			int cur_row = m_column_indices[i];
			if(inB[cur_row]) continue;
			if(c_val[i] > LOC_INFINITY / 2.0) continue;
			double reduced_cval = c_val[i] - u_val[cur_row] - v_val[cur_node];
			if (reduced_cval + 1e-10 < 0) 
				throw system_error(system_error::Negative_MC64_weight, "Negative reduced weight in MC64");

			double d_new = lsp + reduced_cval;
			if(d_new < lsap) {
				if(!matched[cur_row]) {
					lsap = d_new;
					isap = cur_row;
					ksap = i;

					match_nodes[isap] = cur_node;
				} else if (d_new < d_vals[cur_row]){
					d_vals[cur_row] = d_new;
					prev[match_nodes[cur_row]] = cur_node;
					Q.push(thrust::make_tuple(cur_row, d_new));
					irn[cur_row] = i;
				}
			}
		}

		Dijkstra min_d;
		bool found = false;

		while(!Q.empty()) {
			min_d = Q.top();
			Q.pop();
			if(visited[thrust::get<0>(min_d)]) 
				continue;
			found = true;
			break;
		}
		if(!found)
			break;

		int tmp_idx = thrust::get<0>(min_d);
		visited[tmp_idx] = true;

		lsp = thrust::get<1>(min_d);
		if(lsap <= lsp) {
			visited[tmp_idx] = false;
			d_vals[tmp_idx] = LOC_INFINITY;
			break;
		}
		inB[tmp_idx] = true;
		B[b_cnt++] = tmp_idx;

		cur_node = match_nodes[tmp_idx];
	}

	if(lsap < LOC_INFINITY / 2.0) {
		matched[isap] = true;
		cur_node = match_nodes[isap];

		v_val[cur_node] = c_val[ksap];

		while(prev[cur_node] >= 0) {
			match_nodes[isap] = cur_node;

			int next_ksap = rev_match_nodes[cur_node];
			int next_isap = m_column_indices[next_ksap];
			next_ksap = irn[next_isap];

			rev_match_nodes[cur_node] = ksap;

			cur_node = prev[cur_node];
			isap = next_isap;
			ksap = next_ksap;
		}
		match_nodes[isap] = cur_node;
		rev_match_nodes[cur_node] = ksap;
		rev_matched[cur_node] = true;

		for (i = 0; i < b_cnt; i++) {
			int tmp_row = B[i];
			int j_val = match_nodes[tmp_row];
			int tmp_k = rev_match_nodes[j_val];
			u_val[tmp_row] += d_vals[tmp_row] - lsap;
			v_val[j_val] = c_val[tmp_k] - u_val[tmp_row];
			d_vals[tmp_row] = LOC_INFINITY;
			visited[tmp_row] = false;
			inB[tmp_row] = false;
		}

		while(!Q.empty()) {
			Dijkstra tmpD = Q.top();
			Q.pop();
			d_vals[thrust::get<0>(tmpD)] = LOC_INFINITY;
		}
	}
}

void
NewMC64::print(std::ostream& o)
{
	o << "Dimension: "<<m_n << " NNZ: " << m_nnz << std::endl;

	o << "Row offsets: " << std::endl;
	for (int i = 0; i <= m_n; i++)
		o << m_row_offsets[i] << " ";
	o << std::endl;

	o << "Column indices: " << std::endl;
	for (int i = 0; i < m_nnz; i++)
		o << m_column_indices[i] << " ";
	o << std::endl;

	o << "Values: " << std::endl;
	for (int i = 0; i < m_nnz; i++)
		o << m_values[i] << " ";
	o << std::endl;

	if (m_done)
	{
		o << "Row permutation: " << std::endl;
		IntVector& rowPerm  = m_rowPerm;
		for (int i = 0; i < m_n; i++)
			o << rowPerm[i] << " ";
		o << std::endl;
	}

	if (m_done)
	{
		o << "Row Scale: " << std::endl;
		DoubleVector& rowScale = m_rowScale;
		for (int i = 0; i < m_n; i++)
			o << rowScale[i] << " ";
		o << std::endl;
	}

	if (m_done)
	{
		o << "Column Scale: " << std::endl;
		DoubleVector& colScale = m_colScale;
		for (int i = 0; i < m_n; i++)
			o << colScale[i] << " ";
		o << std::endl;
	}
	if (m_done)
		o << std::endl;
}

} // end namespace mc64

#endif
