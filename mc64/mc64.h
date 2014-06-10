#ifndef MC64_H
#define MC64_H

#include <vector>
#include <queue>
#include <cmath>
#include <iostream>
#include <cstdio>
#include <algorithm>

#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/gather.h>
#include <thrust/logical.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/binary_search.h>
#include <thrust/system/cuda/execution_policy.h>

#include "mc64/common.h"
#include "mc64/timer.h"
#include "mc64/exception.h"
#include "mc64/device/kernels.cuh"

namespace mc64  {

// -----------------------------------------------------------------------------
// Base class for the MC64 algorithms
// -----------------------------------------------------------------------------
class MC64_base
{
protected:
  typedef typename thrust::tuple<int, double>     Dijkstra;

  double         m_time_pre;
  double         m_time_first;
  double         m_time_second;
  double         m_time_post;
  double         m_time_total;

  size_t         m_n;
  size_t         m_nnz;

  bool           m_done;

  template <typename IVector>
  static void offsets_to_indices(const IVector& offsets, IVector& indices)
  {
    // convert compressed row offsets into uncompressed row indices
    thrust::fill(indices.begin(), indices.end(), 0);
    thrust::scatter( thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(offsets.size()-1),
        offsets.begin(),
        indices.begin());
    thrust::inclusive_scan(indices.begin(), indices.end(), indices.begin(), thrust::maximum<int>());
  }

  template <typename IVector>
  static void indices_to_offsets(const IVector& indices, IVector& offsets)
  {
    // convert uncompressed row indices into compressed row offsets
    thrust::lower_bound(indices.begin(),
        indices.end(),
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(offsets.size()),
        offsets.begin());
  }

  static const double LOC_INFINITY;

public:
  MC64_base(): m_done(false) {}
  virtual ~MC64_base() {}

  virtual void execute(bool scale = true) = 0;

  virtual void print(std::ostream &o) = 0;

  template <typename T>
  struct AbsoluteValue
  {
    inline
    __host__ __device__
    T operator() (T a) {return (a < 0 ? (-a) : a);}
  };

  struct ClearValue: public thrust::unary_function<double, double>
  {
    inline
    __host__ __device__
    double operator() (double a)
    {
      return 0.0;
    }
  };

  template <typename VType>
  struct CompareValue
  {
    inline
    __host__ __device__
    bool operator () (const thrust::tuple<int, VType> &a, const thrust::tuple<int, VType> &b) const {return thrust::get<1>(a) > thrust::get<1>(b);}
  };

  struct Exponential: public thrust::unary_function<double, double>
  {
    inline
    __host__ __device__
    double operator() (double a)
    {
      return exp(a);
    }
  };

  double getTimeTotal() const {return m_time_total;}
  double getTimePre() const {return m_time_pre;}
  double getTimeFirst() const {return m_time_first;}
  double getTimeSecond() const {return m_time_second;}
  double getTimePost() const {return m_time_post;}
};

const double MC64_base::LOC_INFINITY = 1e37;

// -----------------------------------------------------------------------------
// MC64 algorithm using "standard" CUDA (i.e., no unified memory)
// -----------------------------------------------------------------------------
class MC64: public MC64_base
{
private:
  typedef typename thrust::host_vector<int>       IntVectorH;
  typedef typename thrust::device_vector<int>     IntVectorD;
  typedef typename thrust::host_vector<double>    DoubleVectorH;
  typedef typename thrust::device_vector<double>  DoubleVectorD;
  typedef typename thrust::host_vector<bool>      BoolVectorH;
  typedef typename thrust::device_vector<bool>    BoolVectorD;

  IntVectorH     m_row_offsets;
  IntVectorH     m_column_indices;
  DoubleVectorH  m_values;

  IntVectorD     m_rowPerm;
  DoubleVectorD  m_rowScale;
  DoubleVectorD  m_colScale;

  void formBipartiteGraph(DoubleVectorD &d_c_val, DoubleVectorD &d_max_val_in_col);

  void initPartialMatch(DoubleVectorH& c_val,
              DoubleVectorH& rowScale,
              DoubleVectorH& colScale,
              IntVectorH&    rowReordering,
              IntVectorH&    rev_match_nodes,
              BoolVectorH&   matched,
              BoolVectorH&   rev_matched);

  void findShortestAugPath(int             init_node,
               BoolVectorH&    matched,
               BoolVectorH&    rev_matched,
               IntVectorH&     match_nodes,
               IntVectorH&     rev_match_nodes,
               IntVectorH&     prev,
               DoubleVectorH&  u_val,
               DoubleVectorH&  v_val,
               DoubleVectorH&  c_val,
               IntVectorH&     irn);

public:
  MC64(const IntVectorH&    row_offsets,
       const IntVectorH&    column_indices,
       const DoubleVectorH& values)
  : m_row_offsets(row_offsets),
    m_column_indices(column_indices),
    m_values(values)
  {
    int n = row_offsets.size() - 1;
    m_rowPerm.resize(n);
    m_n   = n;
    m_nnz = m_values.size();
  }

  ~MC64() {}

  void execute(bool scale = true);
  void print(std::ostream &o);
};

// -----------------------------------------------------------------------------
// Implementation of MC64 functions
// -----------------------------------------------------------------------------

void
MC64::execute(bool scale)
{
  DoubleVectorD d_c_val;
  DoubleVectorD d_max_val_in_col(m_n, 0);

  CPUTimer total_timer;
  GPUTimer gpu_timer;
  CPUTimer cpu_timer;

  total_timer.Start();
  gpu_timer.Start();
  formBipartiteGraph(d_c_val, d_max_val_in_col);
  gpu_timer.Stop();
  m_time_pre = gpu_timer.getElapsed();

  cpu_timer.Start();
  DoubleVectorH c_val    =  d_c_val;
  DoubleVectorH rowScale(m_n);
  DoubleVectorH colScale(m_n);
  IntVectorH    rowReordering(m_n);
  IntVectorH    rev_match_nodes(m_nnz);
  BoolVectorH   matched(m_n, 0);
  BoolVectorH   rev_matched(m_n, 0);
  initPartialMatch(c_val, colScale, rowScale, rowReordering, rev_match_nodes, matched, rev_matched);
  cpu_timer.Stop();
  m_time_first = cpu_timer.getElapsed();

  cpu_timer.Start();
  {
    IntVectorH    irn(m_n);
    IntVectorH    prev(m_n);
    for (int i = 0; i < m_n; i++) {
      if(rev_matched[i]) continue;
      findShortestAugPath(i, matched, rev_matched, rowReordering, rev_match_nodes,  prev, colScale, rowScale, c_val, irn);
    }

    if (thrust::any_of(matched.begin(), matched.end(), thrust::logical_not<bool>())) 
      throw system_error(system_error::Matrix_singular, "Singular matrix found\n");

    DoubleVectorH max_val_in_col  = d_max_val_in_col;
    thrust::transform(colScale.begin(), colScale.end(), colScale.begin(), Exponential());
    thrust::transform(thrust::make_transform_iterator(rowScale.begin(), Exponential()),
        thrust::make_transform_iterator(rowScale.end(), Exponential()),
        max_val_in_col.begin(),
        rowScale.begin(),
        thrust::divides<double>());

  }
  cpu_timer.Stop();
  m_time_second = cpu_timer.getElapsed();

  gpu_timer.Start();
  m_rowScale = rowScale;
  m_colScale = colScale;

  IntVectorD   d_rowReordering = rowReordering;
  thrust::scatter(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(m_n), d_rowReordering.begin(), m_rowPerm.begin());

  IntVectorH    rowPerm  = m_rowPerm;
  IntVectorH    row_indices(m_nnz);
  DoubleVectorH values(m_nnz);

  if (scale) {
    for (int i = 0; i < m_n; i++) {
      int start_idx = m_row_offsets[i], end_idx = m_row_offsets[i+1];
      int new_row = rowPerm[i];
      for (int l = start_idx; l < end_idx; l++) {
        row_indices[l] = new_row;
        int to   = (m_column_indices[l]);
        double scaleFact = (rowScale[i] * colScale[to]);
        values[l] = scaleFact * m_values[l];
      }
    }
    thrust::copy(values.begin(), values.end(), m_values.begin());
  } else {
    for (int i = 0; i < m_n; i++) {
      int start_idx = m_row_offsets[i], end_idx = m_row_offsets[i+1];
      int new_row = rowPerm[i];
      for (int l = start_idx; l < end_idx; l++)
        row_indices[l] = new_row;
    }
  }

  {
    IntVectorH& row_offsets = m_row_offsets;
    IntVectorH  column_indices(m_nnz);

    thrust::fill(row_offsets.begin(), row_offsets.end(), 0);
    for (int i = 0; i < m_nnz; i++)
      row_offsets[row_indices[i]] ++;

    thrust::inclusive_scan(row_offsets.begin(), row_offsets.end(), row_offsets.begin());

    for (int i = m_nnz - 1; i >= 0; i--) {
      int idx = (--row_offsets[row_indices[i]]);
      column_indices[idx] = m_column_indices[i];
      values[idx]         = m_values[i];
    }

    m_column_indices = column_indices;
    m_values         = values;
  }
  gpu_timer.Stop();
  m_time_post = gpu_timer.getElapsed();

  m_done = true;

  total_timer.Stop();
  m_time_total = total_timer.getElapsed();
}

void
MC64::formBipartiteGraph(DoubleVectorD &d_c_val, DoubleVectorD &d_max_val_in_col)
{
  IntVectorD    d_row_offsets  = m_row_offsets;

  d_c_val  =  m_values;

  {
    IntVectorD    d_row_indices(m_nnz);
    offsets_to_indices(d_row_offsets, d_row_indices);
    thrust::transform(d_c_val.begin(), d_c_val.end(), d_c_val.begin(), AbsoluteValue<double>());
    thrust::reduce_by_key(d_row_indices.begin(), d_row_indices.end(), d_c_val.begin(), thrust::make_discard_iterator(), d_max_val_in_col.begin(), thrust::equal_to<double>(), thrust::maximum<double>());
  }

  double *p_c_val          = thrust::raw_pointer_cast(&d_c_val[0]);
  const int *p_row_offsets = thrust::raw_pointer_cast(&d_row_offsets[0]);
  double *p_max_val        = thrust::raw_pointer_cast(&d_max_val_in_col[0]);

  int blockX = m_n, blockY = 1;
  kernelConfigAdjust(blockX, blockY, MAX_GRID_DIMENSION);
  dim3 grids(blockX, blockY);

  device::getResidualValues<<<grids, 64>>>(m_n, p_c_val, p_max_val, p_row_offsets); 
}

void
MC64::initPartialMatch(DoubleVectorH& c_val,
                       DoubleVectorH& rowScale,
                       DoubleVectorH& colScale,
                       IntVectorH&    rowReordering,
                       IntVectorH&    rev_match_nodes,
                       BoolVectorH&   matched,
                       BoolVectorH&   rev_matched)
{
  {
    IntVectorH row_indices(c_val.size());
    offsets_to_indices(m_row_offsets, row_indices);
    thrust::reduce_by_key(row_indices.begin(), row_indices.end(), c_val.begin(), thrust::make_discard_iterator(), rowScale.begin(), thrust::equal_to<double>(), thrust::minimum<double>());
  }

  thrust::fill(colScale.begin(), colScale.end(), LOC_INFINITY);
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
}

void
MC64::findShortestAugPath(int             init_node,
                          BoolVectorH&    matched,
                          BoolVectorH&    rev_matched,
                          IntVectorH&     match_nodes,
                          IntVectorH&     rev_match_nodes,
                          IntVectorH&     prev,
                          DoubleVectorH&  u_val,
                          DoubleVectorH&  v_val,
                          DoubleVectorH&  c_val,
                          IntVectorH&     irn)
{
  static IntVectorH B(m_n, 0);
  int b_cnt = 0;
  static BoolVectorH inB(m_n, false);

  std::priority_queue<Dijkstra, std::vector<Dijkstra>, CompareValue<double> > Q;

  double lsp = 0.0;
  double lsap = LOC_INFINITY;
  int cur_node = init_node;

  int i;

  int isap = -1;
  int ksap = -1;
  prev[init_node] = -1;

  static DoubleVectorH d_vals(m_n, LOC_INFINITY);
  static BoolVectorH visited(m_n, false);

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
MC64::print(std::ostream& o)
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
    IntVectorH rowPerm  = m_rowPerm;
    for (int i = 0; i < m_n; i++)
      o << rowPerm[i] << " ";
    o << std::endl;
  }

  if (m_done)
  {
    o << "Row Scale: " << std::endl;
    DoubleVectorH rowScale = m_rowScale;
    for (int i = 0; i < m_n; i++)
      o << rowScale[i] << " ";
    o << std::endl;
  }

  if (m_done)
  {
    o << "Column Scale: " << std::endl;
    DoubleVectorH colScale = m_colScale;
    for (int i = 0; i < m_n; i++)
      o << colScale[i] << " ";
    o << std::endl;
  }
  if (m_done)
    o << std::endl;
}


} // end namespace mc64

#endif
