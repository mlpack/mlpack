#include "thread.h"

Mutex Mutex::global;

pthread_mutex_t Mutex::fast_mutex_ =
    PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t Mutex::recursive_mutex_ =
    PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;

// TODO: Blank file
