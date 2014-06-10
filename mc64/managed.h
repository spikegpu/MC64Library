#ifndef MANAGED_H
#define MANAGED_H

#include "mc64/common.h"


namespace mc64 {

template <typename T>
class ManagedVector {
  T*     m_p_array;
  size_t m_size;

public:
  typedef typename thrust::device_ptr<T> iterator;

  ManagedVector(): m_p_array(0), m_size(0) {}
  ManagedVector(size_t n): m_size(n) {
    cudaMallocManaged(&m_p_array, sizeof(T) * n);
  }
  ManagedVector(size_t n, T val): m_size(n) {
    cudaMallocManaged(&m_p_array, sizeof(T) * n);
    thrust::fill(thrust::cuda::par, m_p_array, m_p_array + n, val);
    cudaDeviceSynchronize();
  }
  ManagedVector(const ManagedVector &a): m_size(a.m_size) {
    cudaMallocManaged(&m_p_array, sizeof(T) * a.m_size);
    thrust::copy(thrust::cuda::par, a.m_p_array, a.m_p_array + a.m_size, m_p_array);
    cudaDeviceSynchronize();
  }
  ~ManagedVector() {cudaFree(m_p_array);}

  ManagedVector& operator=(const ManagedVector &a) {
    if (m_size < a.m_size) {
      m_size = a.m_size;
      cudaFree(m_p_array);
      cudaMallocManaged(&m_p_array, sizeof(T) * a.m_size);
      thrust::copy(thrust::cuda::par, a.m_p_array, a.m_p_array + a.m_size, m_p_array);
      cudaDeviceSynchronize();
    } else {
      m_size = a.m_size;
      thrust::copy(thrust::cuda::par, a.m_p_array, a.m_p_array + a.m_size, m_p_array);
      cudaDeviceSynchronize();
    }

    return *this;
  }

  thrust::device_ptr<T> begin() const {return thrust::device_pointer_cast(&m_p_array[0]);}
  thrust::device_ptr<T> end()   const {return thrust::device_pointer_cast(&m_p_array[m_size]);}

  T& operator[](size_t n)    {return m_p_array[n];}
  const T& operator[](size_t n)  const  {return m_p_array[n];}

  size_t size() const {return m_size;}

  void resize(size_t n)  {
    if (m_size >= n) m_size = n;
    else {
      T *p_tmp;
      cudaMallocManaged(&p_tmp, sizeof(T) * n);

      if (m_size > 0) {
        thrust::copy(thrust::cuda::par, m_p_array, m_p_array + m_size, p_tmp);
        cudaDeviceSynchronize();
      }

      m_size = n;
      cudaFree(m_p_array);

      m_p_array = p_tmp;
    }
  }

};


} // namespace mc64


#endif
