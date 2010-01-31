#ifndef YASLI_PLATFORM_H_
#define YASLI_PLATFORM_H_

// $Id: platform.h 754 2006-10-17 19:59:11Z syntheticpp $


// Most conservative

#define YASLI_HAS_EFFICIENT_MSIZE 0
#define YASLI_REALLOC_AFTER_NEW 0
#define YASLI_HAS_EXPAND 0

#include <malloc.h>

// Works on MSVC (all versions)
#if defined(_MSC_VER)

    #if defined(NDEBUG)//why only if ndebug?
        #undef YASLI_REALLOC_AFTER_NEW//this is not used: what is it's intention?
        #define YASLI_REALLOC_AFTER_NEW 1

        #undef YASLI_HAS_EFFICIENT_MSIZE 
        #define YASLI_HAS_EFFICIENT_MSIZE 1
    #endif

    // On Wintel platforms, uninit pointers can be copied
    #define YASLI_UNDEFINED_POINTERS_COPYABLE 1

    namespace yasli_platform 
    {
        inline size_t msize(const void *p) 
        { 
            return _msize(const_cast<void*>(p)); 
        }
    }
    
    #undef YASLI_HAS_EXPAND
    #define YASLI_HAS_EXPAND 1
    namespace yasli_platform 
    {
        inline void* expand(const void *p, size_t s) 
        { 
            return _expand (const_cast<void*>(p), s);
        }
    }
    
#elif defined(__MINGW32_VERSION) && !defined(RC_INVOKED)
//I havinclude Malloc.h in order to find whether this library is Mingw32 
//i.e. defines __MINGW32_VERSION, 
//hmm, I really need to sort this out in earlier versions
//it was _STRICT_ANSI_ not RC_INVOKED but I don't know the details
    #undef YASLI_HAS_EFFICIENT_MSIZE 
    #define YASLI_HAS_EFFICIENT_MSIZE 1
    namespace yasli_platform 
    {
        inline size_t msize(const void *p) 
        { 
            return _msize(const_cast<void*>(p)); 
        }
    }
    
    #undef YASLI_HAS_EXPAND 
    #define YASLI_HAS_EXPAND 1
    namespace yasli_platform 
    {
        inline void* expand(const void *p, size_t s) 
        { 
            return _expand (const_cast<void*>(p), s);
        }
    }
    
    #define YASLI_UNDEFINED_POINTERS_COPYABLE 1 //well it appears to compile ok anyway
#endif





#endif // YASLI_PLATFORM_H_
