#include <iostream>

#define LOG_EVENT(a) std::cout<<a<<std::endl;
#define CUDA_FREE(ptr) if((ptr) != NULL) { cudaFree(ptr); ptr = NULL; }
#define CUDA_CALL(func) checkCudaErrors(func)
