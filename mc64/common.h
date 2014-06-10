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
# include <thrust/device_ptr.h>
# include <thrust/system/cuda/execution_policy.h>


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


} // namespace spike


#endif
