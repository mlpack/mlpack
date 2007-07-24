#include "thread.h"

Mutex Mutex::global;

#ifdef DEBUG
pthread_mutex_t Mutex::fast_mutex_ =
    PTHREAD_ERRORCHECK_MUTEX_INITIALIZER_NP;
#else
pthread_mutex_t Mutex::fast_mutex_ =
    PTHREAD_MUTEX_INITIALIZER;
#endif
pthread_mutex_t Mutex::recursive_mutex_ =
    PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;

// TODO: Blank file
