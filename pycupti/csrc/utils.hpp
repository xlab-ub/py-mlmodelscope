#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <stddef.h> 
#include <stdint.h>
#include <stdbool.h>

#if defined(__linux__) 
#define DllExport   
#else 
#define DllExport   __declspec( dllexport )
#endif 

#ifdef __cplusplus
extern "C" {
#else
#endif  /* __cplusplus */
DllExport void bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords); 
DllExport void freebufferRequested(uint8_t* buffer); 
DllExport void onCallback(int);
DllExport void startProfiling(char*);
DllExport double* endProfiling(uint64_t*);
DllExport bool getProfilingAPI(void);
#ifdef __cplusplus
}
#endif  /* __cplusplus */

#endif /* __UTILS_HPP__ */

