
#ifndef MLPACK_BINDINGS_GO_MLPACK_SERIALIZATION_UTIL_H
#define MLPACK_BINDINGS_GO_MLPACK_SERIALIZATION_UTIL_H

#include <mlpack/core.hpp>

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

extern const char* MLPACK_SerializeOut(const char *ptr, const char *name);

extern void MLPACK_SerializeIn(const char *ptr, const char *str, const char *name);

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif

#endif
