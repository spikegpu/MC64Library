#ifndef COMMON_H
#define COMMON_H



#define ALWAYS_ASSERT

#ifdef WIN32
typedef long long int64_t;
#endif



// ----------------------------------------------------------------------------
// If ALWAYS_ASSERT is defined, we make sure that  assertions are triggered 
// even if NDEBUG is defined.
// ----------------------------------------------------------------------------
#ifdef ALWAYS_ASSERT
// If NDEBUG is actually defined, remember this so
// we can restore it.
#  ifdef NDEBUG
#    define NDEBUG_ACTIVE
#    undef NDEBUG
#  endif
// Include the assert.h header file here so that it can
// do its stuff while NDEBUG is guaranteed to be disabled.
#  include <assert.h>
// Restore NDEBUG mode if it was active.
#  ifdef NDEBUG_ACTIVE
#    define NDEBUG
#    undef NDEBUG_ACTIVE
#  endif
#else
// Include the assert.h header file using whatever the
// current definition of NDEBUG is.
#  include <assert.h>
#endif

# include <memory.h>


// ----------------------------------------------------------------------------


namespace mc64 {

const unsigned int BLOCK_SIZE = 512;

const unsigned int MAX_GRID_DIMENSION = 32768;

inline
void kernelConfigAdjust(int &numThreads, int &numBlockX, const int numThreadsMax) {
	if (numThreads > numThreadsMax) {
		numBlockX = (numThreads + numThreadsMax - 1) / numThreadsMax;
		numThreads = numThreadsMax;
	}
}

inline
void kernelConfigAdjust(int &numThreads, int &numBlockX, int &numBlockY, const int numThreadsMax, const int numBlockXMax) {
	kernelConfigAdjust(numThreads, numBlockX, numThreadsMax);
	if (numBlockX > numBlockXMax) {
		numBlockY = (numBlockX + numBlockXMax - 1) / numBlockXMax;
		numBlockX = numBlockXMax;
	}
}

template <typename T>
class ManagedVector {
	T*     m_p_array;
	size_t m_size;

public:
	ManagedVector(): m_p_array(0), m_size(0) {}
	ManagedVector(size_t n): m_size(n) {
		cudaMallocManaged(&m_p_array, sizeof(T) * n);
	}
	ManagedVector(size_t n, T val): m_size(n) {
		cudaMallocManaged(&m_p_array, sizeof(T) * n);
		for (int i = 0; i < n; i++)
			m_p_array[i] = val;
	}
	ManagedVector(const ManagedVector &a): m_size(a.m_size) {
		cudaMallocManaged(&m_p_array, sizeof(T) * a.m_size);
		memcpy(m_p_array, a.m_p_array, sizeof(T) * a.m_size);
	}
	~ManagedVector() {cudaFree(m_p_array);}

	ManagedVector& operator=(const ManagedVector &a) {
		m_size = a.m_size;
		cudaMallocManaged(&m_p_array, sizeof(T) * a.m_size);
		memcpy(m_p_array, a.m_p_array, sizeof(T) * a.m_size);

		return *this;
	}

	T *begin() const {return m_p_array;}
	T *end()   const {return m_p_array + m_size;}

	T& operator[](size_t n)    {return m_p_array[n];}
	const T& operator[](size_t n)  const  {return m_p_array[n];}

	size_t size() const {return m_size;}

	void resize(size_t n)  {
		if (m_size >= n) m_size = n;
		else {
			T *p_tmp;
			cudaMallocManaged(&p_tmp, sizeof(T) * n);

			if (m_size > 0)
				memcpy(p_tmp, m_p_array, sizeof(T) * m_size);

			m_size = n;
			cudaFree(m_p_array);

			m_p_array = p_tmp;
		}
	}

};


} // namespace spike


#endif
