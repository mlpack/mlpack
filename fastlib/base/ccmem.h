// Copyright 2007 Georgia Institute of Technology. All rights reserved.
/**
 * @file ccmem.h
 *
 * Low-level (read: scary) memory management routines used by core
 * data structures.
 *
 * @see namespace mem
 */

#ifndef BASE_CCMEM_H
#define BASE_CCMEM_H

#include "common.h"
#include "debug.h"

#include <new>

/**
 * Wrappers and tools for low-level memory management, including:
 *
 * @li debuggable memory allocation wrappers
 * @li poisoning, zeroing, copying, and swapping of memory
 * @li construction and destruction of allocated object arrays
 * @li absolute pointer arithmetic functions
 *
 * You likely do not need to care about these functions: use new and
 * delete (like normal) for allocation of single objects and FASTlib's
 * ArrayList (or Vector or Matrix) for arrays.
 *
 * If you really need to manage your own memory, use these instead of
 * malloc and free, because these will perform "memory poising" in
 * debug mode.
 */
namespace mem {
  /** Fills memory with BIG_BAD_NUMBER, measured in bytes. */
  template<typename T>
  inline T *DebugPoisonBytes(T *array, size_t bytes) {
#ifdef DEBUG
    int32 *ptr = reinterpret_cast<int32 *>(array);
    size_t len = bytes / sizeof(int32);
    for (size_t i = 0; i < len; ++i) {
      ptr[i] = BIG_BAD_NUMBER;
    }
#endif
    return array;
  }
  /** Fills memory with BIG_BAD_NUMBER, measured in elements. */
  template<typename T>
  inline T *DebugPoison(T *array, size_t elems = 1) {
    return DebugPoisonBytes(array, elems * sizeof(T));
  }
  /** Allocates a (debug) poisoned array, measured in bytes. */
  template<typename T>
  inline T *AllocBytes(size_t bytes) {
#ifdef SCALE_NORMAL
    /* Sanity check for small-scale problems. */
    DEBUG_BOUNDS(bytes, BIG_BAD_NUMBER);
#endif
    return DebugPoisonBytes(reinterpret_cast<T *>(::malloc(bytes)), bytes);
  }
  /** Allocates a (debug) poisoned array, measured in elements. */
  template<typename T>
  inline T *Alloc(size_t elems = 1) {
#ifdef SCALE_NORMAL
    /* Sanity check for small-scale problems. */
    DEBUG_BOUNDS(elems, BIG_BAD_NUMBER);
#endif
    return AllocBytes<T>(elems * sizeof(T));
  }

  /** Bit-zeros memory, measured in bytes. */
  template<typename T>
  inline T *BitZeroBytes(T *array, size_t bytes) {
    return reinterpret_cast<T *>(::memset(array, 0, bytes));
  }
  /** Bit-zeros memory, measured in elements. */
  template<typename T>
  inline T *BitZero(T *array, size_t elems = 1) {
    return BitZeroBytes(array, elems * sizeof(T));
  }
  /** Allocates a bit-zerod array, measured in bytes. */
  template<typename T>
  inline T *AllocBitZeroBytes(size_t bytes) {
    return reinterpret_cast<T *>(::calloc(bytes, 1));
  }
  /** Allocates a bit-zerod array, measured in elements. */
  template<typename T>
  inline T *AllocBitZero(size_t elems = 1) {
    return reinterpret_cast<T *>(::calloc(elems, sizeof(T)));
  }

  /** Bit-copies from src to dest, measured in bytes. */
  template<typename T>
  inline T *BitCopyBytes(T *dest, const T *src, size_t bytes) {
    return reinterpret_cast<T *>(::memcpy(dest, src, bytes));
  }
  /** Bit-copies from src to dest, measured in elements. */
  template<typename T>
  inline T *BitCopy(T *dest, const T *src, size_t elems = 1) {
    return BitCopyBytes(dest, src, elems * sizeof(T));
  }
  /** Allocates an array bit-copied from src, measured in bytes. */
  template<typename T>
  inline T *AllocBitCopyBytes(const T *src, size_t bytes) {
    return BitCopyBytes(reinterpret_cast<T *>(::malloc(bytes)), src, bytes);
  }
  /** Allocates an array bit-copied from src, measured in elements. */
  template<typename T>
  inline T *AllocBitCopy(const T *src, size_t elems = 1) {
    return AllocBitCopyBytes(src, elems * sizeof(T));
  }

  /**
   * Resizes allocated memory, mesured in bytes.
   *
   * Added bytes (if any) are not poisoned or zeroed.  The input
   * pointer is invalidated and should be replaced by the return in
   * all subsequent uses.
   */
  template<typename T>
  inline T *ReallocBytes(T *array, size_t bytes) {
     return reinterpret_cast<T *>(::realloc(array, bytes));
  }
  /**
   * Resizes allocated memory, measured in elements.
   *
   * Added elements (if any) are not poisoned or zeroed.  The input
   * pointer is invalidated and should be replaced by the return in
   * all subsequent uses.
   */
  template<typename T>
  inline T *Realloc(T *array, size_t elems) {
     return ReallocBytes<T>(array, elems * sizeof(T));
  }

  /** Frees memory allocated by mem::Alloc and its derivatives. */
  template<typename T>
  inline void Free(T* ptr) {
     ::free(ptr);
  }



  /** Buffer size used when swapping memory via memcpy. */
#define SWAP_BUF_SIZE 64

  /**
   * Bit-swaps two arrays, measured in bytes.
   *
   * This code works best for arrays starting at multiple-of-eight
   * (and higher powers of two) byte locations.  Freshly allocated
   * memory and locations within arrays of longs, doubles, and most
   * structs will have this property.  Suboptimal performance arises
   * when swapping between offset locations in arrays of small types,
   * such as portions of strings.
   */
  template<typename T>
  inline void BitSwapBytes(T *a, T *b, size_t bytes) {
    char *a_cp = reinterpret_cast<char *>(a);
    char *b_cp = reinterpret_cast<char *>(b);
    char buf[SWAP_BUF_SIZE];

    while (bytes > SWAP_BUF_SIZE) {
      ::memcpy(buf, a_cp, SWAP_BUF_SIZE);
      ::memcpy(a_cp, b_cp, SWAP_BUF_SIZE);
      ::memcpy(b_cp, buf, SWAP_BUF_SIZE);

      bytes -= SWAP_BUF_SIZE;
      a_cp += SWAP_BUF_SIZE;
      b_cp += SWAP_BUF_SIZE;
    }
    if (bytes > 0) {
      ::memcpy(buf, a_cp, bytes);
      ::memcpy(a_cp, b_cp, bytes);
      ::memcpy(b_cp, buf, bytes);
    }
  }
  /**
   * Bit-swaps two arrays, measured in elements.
   *
   * This code is optimized for swapping arrays starting at
   * multiple-of-eight byte locations.  Freshly allocated memory and
   * all locations within arrays of longs, doubles, and most structs
   * will have this property.  Suboptimal performance will arise only
   * when swapping between offset locations in arrays of small types,
   * such as portions of strings.
   */
  template<typename T>
  inline void BitSwap(T *a, T *b, size_t elems = 1) {
    BitSwapBytes(a, b, elems * sizeof(T));
  }



  /** Default constructs an element. */
  template<typename T>
  inline T *Construct(T *ptr) {
    new(ptr) T();
    return ptr;
  }
  /** Default constructs each element in an array. */
  template<typename T>
  inline T *Construct(T *array, size_t elems) {
    for (size_t i = 0; i < elems; ++i) {
      new(array + i) T();
    }
    return array;
  }
  /** Destructs an element. */
  template<typename T>
  inline T *Destruct(T *ptr) {
    ptr->~T();
    return DebugPoison(ptr);
  }
  /** Destructs each element in an array. */
  template<typename T>
  inline T *Destruct(T *array, size_t elems) {
    for (size_t i = 0; i < elems; ++i) {
      array[i].~T();
    }
    return DebugPoison(array, elems);
  }
  /** Element-wise copy constructs one element given another. */
  template<typename T, typename U>
  inline T *CopyConstruct(T *dest, const U *src) {
    new(dest) T(*src);
    return dest;
  }
  /** Element-wise copy constructs one array given another. */
  template<typename T, typename U>
  inline T *CopyConstruct(T *dest, const U *src, size_t elems) {
    for (size_t i = 0; i < elems; ++i) {
      new(dest + i) T(src[i]);
    }
    return dest;
  }

  /** Simple constructors and destcutors for primatives types. */
#define BASE_CCMEM__SIMPLE_CONSTRUCTORS(T) \
  template<> \
  inline T *Construct< T >(T *ptr) \
    {return DebugPoison(ptr);} \
  template<> \
  inline T *Construct< T >(T *array, size_t elems) \
    {return DebugPoison(array, elems);} \
  template<> \
  inline T *Destruct< T >(T *ptr) \
    {return DebugPoison(ptr);} \
  template<> \
  inline T *Destruct< T >(T *array, size_t elems) \
    {return DebugPoison(array, elems);} \
  template<> \
  inline T *CopyConstruct< T >(T *dest, const T *src) \
    {return BitCopy(dest, src, 1);} \
  template<> \
  inline T *CopyConstruct< T >(T *dest, const T *src, size_t elems) \
    {return BitCopy(dest, src, elems);}

  BASE_CCMEM__SIMPLE_CONSTRUCTORS(char)
  BASE_CCMEM__SIMPLE_CONSTRUCTORS(short)
  BASE_CCMEM__SIMPLE_CONSTRUCTORS(int)
  BASE_CCMEM__SIMPLE_CONSTRUCTORS(long)
  BASE_CCMEM__SIMPLE_CONSTRUCTORS(long long)
  BASE_CCMEM__SIMPLE_CONSTRUCTORS(unsigned char)
  BASE_CCMEM__SIMPLE_CONSTRUCTORS(unsigned short)
  BASE_CCMEM__SIMPLE_CONSTRUCTORS(unsigned int)
  BASE_CCMEM__SIMPLE_CONSTRUCTORS(unsigned long)
  BASE_CCMEM__SIMPLE_CONSTRUCTORS(unsigned long long)
  BASE_CCMEM__SIMPLE_CONSTRUCTORS(float)
  BASE_CCMEM__SIMPLE_CONSTRUCTORS(double)

  /** No-op constructs an array of pointers. */
  template<typename T>
  inline T **Construct(T **array, size_t elems = 1)
    {return DebugPoison(array, elems);}
  /** No-op destructs an array of pointers. */
  template<typename T>
  inline T **Destruct(T **array, size_t elems = 1)
    {return DebugPoison(array, elems);}
  /** Bit-copy copy constructs an array of pointers. */
  template<typename T>
  inline T **CopyConstruct(T **dest, const T **src, size_t elems = 1)
    {return BitCopy(dest, src, elems);}

  /** Constructs each element in an array with an initial value. */
  template<typename T, typename U>
  inline T *RepeatConstruct(T *array, const U &init, size_t elems) {
    for (size_t i = 0; i < elems; ++i) {
      new(array + i) T(init);
    }
    return array;
  }

  /** Allocates and default constructs an array. */
  template<typename T>
  inline T *AllocConstruct(size_t elems = 1) {
    return Construct(Alloc<T>(elems), elems);
  }
  /** Allocates and element-wise copy constructs an array. */
  template<typename T, typename U>
  inline T *AllocCopyConstruct(const U *src, size_t elems = 1) {
    return CopyConstruct(Alloc<T>(elems), src, elems);
  }
  /** Allocates and copy constructs an array. */
  template<typename T, typename U>
  inline T *AllocRepeatConstruct(const U &init, size_t elems) {
    return RepeatConstruct(Alloc<T>(elems), init, elems);
  }

  /** Destructs and frees an array. */
  template<typename T>
  inline void FreeDestruct(T *array, size_t elems = 1) {
    Free(Destruct(array, elems));
  }



  /** Offsets a pointer by a given number of bytes. */
  template<typename T>
  inline T *PtrAddBytes(const T *ptr, ptrdiff_t bytes) {
    /* Const cast to prevent compilation errors for const T. */
    return reinterpret_cast<T *>(const_cast<char *>(
      reinterpret_cast<const char *>(ptr) + bytes));
  }
  /** Finds the byte difference of two pointers, i.e. lhs - rhs. */
  template<typename T, typename U>
  inline ptrdiff_t PtrDiffBytes(const T *lhs, const U *rhs) {
    return reinterpret_cast<const char *>(lhs)
      - reinterpret_cast<const char *>(rhs);
  }
  /** Converts a pointer to its integral absolute address. */
  template<typename T>
  inline ptrdiff_t PtrAbsAddr(const T *ptr) {
    return reinterpret_cast<ptrdiff_t>(ptr);
  }
  /** Determines if two pointers are the same, i.e. lhs == rhs. */
  template<typename T, typename U>
  inline bool PtrsEqual(const T *lhs, const U *rhs) {
    return reinterpret_cast<size_t>(lhs) == reinterpret_cast<size_t>(rhs);
  } 
};

#endif
