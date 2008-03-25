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
#include "cc.h"

#include <new>

#define MEM__DEBUG_MEMORY(ptr) \
    DEBUG_ASSERT_MSG((ptr) != NULL, "out of memory")

namespace mem__private {
  const size_t BIG_BAD_BUF_SIZE = 64;

  extern const int32 BIG_BAD_BUF[];

  void PoisonBytes(char *array_cp, size_t bytes);

  const size_t SWAP_BUF_SIZE = 64;

  void SwapBytes(char *a_cp, char *b_cp, size_t bytes);
};

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
  T *PoisonBytes(T *array, size_t bytes) {
    char *array_cp = reinterpret_cast<char *>(array);
    mem__private::PoisonBytes(array_cp, bytes);
    return array;
  }
  /** Fills memory with BIG_BAD_NUMBER, measured in elements. */
  template<typename T>
  inline T *Poison(T *array, size_t elems) {
    return PoisonBytes(array, elems * sizeof(T));
  }
  /** Fills an element with BIG_BAD_NUMBER. */
  template<typename T>
  inline T *Poison(T *ptr) {
    if (sizeof(T) <= mem__private::BIG_BAD_BUF_SIZE) {
      return reinterpret_cast<T *>(
          ::memcpy(ptr, mem__private::BIG_BAD_BUF, sizeof(T)));
    } else {
      return PoisonBytes(ptr, sizeof(T));
    }
  }

  /** Fills memory with BIG_BAD_NUMBER, measured in bytes. */
  template<typename T>
  inline T *DebugPoisonBytes(T *array, size_t bytes) {
    DEBUG_ONLY(PoisonBytes(array, bytes));
    return array;
  }
  /** Fills memory with BIG_BAD_NUMBER, measured in elements. */
  template<typename T>
  inline T *DebugPoison(T *array, size_t elems) {
    DEBUG_ONLY(Poison(array, elems));
    return array;
  }
  /** Fills an element with BIG_BAD_NUMBER. */
  template<typename T>
  inline T *DebugPoison(T *ptr) {
    DEBUG_ONLY(Poison(ptr));
    return ptr;
  }

  /** Allocates a (debug) poisoned array, measured in bytes. */
  template<typename T>
  inline T *AllocBytes(size_t bytes) {
#ifdef SCALE_NORMAL
    // sanity check for small-scale problems
    DEBUG_BOUNDS(bytes, BIG_BAD_NUMBER);
#endif
    T *array = reinterpret_cast<T *>(::malloc(bytes));
    MEM__DEBUG_MEMORY(array);
    return DebugPoisonBytes(array, bytes);
  }
  /** Allocates a (debug) poisoned array, measured in elements. */
  template<typename T>
  inline T *Alloc(size_t elems) {
#ifdef SCALE_NORMAL
    // sanity check for small-scale problems
    DEBUG_BOUNDS(elems, BIG_BAD_NUMBER);
#endif
    return AllocBytes<T>(elems * sizeof(T));
  }
  /** Allocates a (debug) poisoned element. */
  template<typename T>
  inline T *Alloc() {
    T *array = reinterpret_cast<T *>(::malloc(sizeof(T)));
    MEM__DEBUG_MEMORY(array);
    return DebugPoisonBytes(array);
  }

  /** Bit-zeros memory, measured in bytes. */
  template<typename T>
  inline T *ZeroBytes(T *array, size_t bytes) {
    return reinterpret_cast<T *>(::memset(array, 0, bytes));
  }
  /** Bit-zeros memory, measured in elements. */
  template<typename T>
  inline T *Zero(T *array, size_t elems = 1) {
    return ZeroBytes(array, elems * sizeof(T));
  }
  /** Allocates a bit-zerod array, measured in bytes. */
  template<typename T>
  inline T *AllocZeroBytes(size_t bytes) {
    T *array = reinterpret_cast<T *>(::calloc(bytes, 1));
    MEM__DEBUG_MEMORY(array);
    return array;
  }
  /** Allocates a bit-zerod array, measured in elements. */
  template<typename T>
  inline T *AllocZero(size_t elems = 1) {
    T *array = reinterpret_cast<T *>(::calloc(elems, sizeof(T)));
    MEM__DEBUG_MEMORY(array);
    return array;
  }

  /** Bit-copies from src to dest, measured in bytes. */
  template<typename T, typename U>
  inline T *CopyBytes(T *dest, const U *src, size_t bytes) {
    return reinterpret_cast<T *>(::memcpy(dest, src, bytes));
  }
  /** Bit-copies from src to dest, measured in elements. */
  template<typename V, typename T, typename U>
  inline T *Copy(T *dest, const U *src, size_t elems = 1) {
    return CopyBytes(dest, src, elems * sizeof(V));
  }
  template<typename T>
  inline T *Copy(T *dest, const T *src, size_t elems = 1) {
    return Copy<T, T, T>(dest, src, elems);
  }
  /** Allocates an array bit-copied from src, measured in bytes. */
  template<typename T, typename U>
  inline T *AllocCopyBytes(const U *src, size_t bytes) {
    T *array = reinterpret_cast<T *>(::malloc(bytes));
    MEM__DEBUG_MEMORY(array);
    return CopyBytes(array, src, bytes);
  }
  /** Allocates an array bit-copied from src, measured in elements. */
  template<typename T, typename U>
  inline T *AllocCopy(const U *src, size_t elems = 1) {
    return AllocCopyBytes<T>(src, elems * sizeof(T));
  }
  template<typename T>
  inline T *AllocCopy(const T *src, size_t elems = 1) {
    return AllocCopy<T, T>(src, elems);
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
    array = reinterpret_cast<T *>(::realloc(array, bytes));
    MEM__DEBUG_MEMORY(array);
    return array;
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
  template<typename T, typename U>
  void SwapBytes(T *a, U *b, size_t bytes) {
    char *a_cp = reinterpret_cast<char *>(a);
    char *b_cp = reinterpret_cast<char *>(b);
    mem__private::SwapBytes(a_cp, b_cp, bytes);
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
  template<typename V, typename T, typename U>
  inline void Swap(T *a, U *b, size_t elems) {
    SwapBytes(a, b, elems * sizeof(V));
  }
  template<typename T>
  inline void Swap(T *a, T *b, size_t elems) {
    Swap<T, T, T>(a, b, elems);
  }
  template<typename V, typename T, typename U>
  inline void Swap(T *a, U *b) {
    if (sizeof(V) <= mem__private::SWAP_BUF_SIZE * 2) {
      char buf[sizeof(V)];

      ::memcpy(buf, a, sizeof(V));
      ::memcpy(a, b, sizeof(V));
      ::memcpy(b, buf, sizeof(V));
    } else {
      SwapBytes(a, b, sizeof(V));
    }
  }
  template<typename T>
  inline void Swap(T *a, T *b) {
    Swap<T, T, T>(a, b);
  }

  /** Bit-moves bytes from src to dest, permitting overlap. */
  template<typename T, typename U>
  inline T *MoveBytes(T *dest, const U *src, size_t bytes) {
    return reinterpret_cast<T *>(::memmove(dest, src, bytes));
  }
  /** Bit-moves elements from src to dest, permitting overlap. */
  template<typename V, typename T, typename U>
  inline T *Move(T *dest, const U *src, size_t elems = 1) {
    return MoveBytes(dest, src, elems * sizeof(V));
  }
  template<typename T>
  inline T *Move(T *dest, const T *src, size_t elems = 1) {
    return Move<T, T, T>(dest, src, elems);
  }



  /** Default Constructs An element. */
  template<typename T>
  inline T *Construct(T *ptr) {
    new(ptr) T;
    return ptr;
  }
  /** Default constructs each element in an array. */
  template<typename T>
  inline T *Construct(T *array, size_t elems) {
    new(array) T[elems];
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
#define BASE_CCMEM__SIMPLE_CONSTRUCTORS(T, TF) \
  template<> \
  inline T *Construct< T >(T *ptr) { \
    return DebugPoison(ptr); \
  } \
  template<> \
  inline T *Construct< T >(T *array, size_t elems) { \
    return DebugPoison(array, elems); \
  } \
  template<> \
  inline T *Destruct< T >(T *ptr) { \
    return DebugPoison(ptr); \
  } \
  template<> \
  inline T *Destruct< T >(T *array, size_t elems) { \
    return DebugPoison(array, elems); \
  } \
  template<> \
  inline T *CopyConstruct< T >(T *dest, const T *src) { \
    return Copy(dest, src, 1); \
  } \
  template<> \
  inline T *CopyConstruct< T >(T *dest, const T *src, size_t elems) { \
    return Copy(dest, src, elems); \
  }

  FOR_ALL_PRIMITIVES_DO(BASE_CCMEM__SIMPLE_CONSTRUCTORS)

  /** No-op constructs an array of pointers. */
  template<typename T>
  inline T **Construct(T **array, size_t elems = 1) {
    return DebugPoison(array, elems);
  }
  /** No-op destructs an array of pointers. */
  template<typename T>
  inline T **Destruct(T **array, size_t elems = 1) {
    return DebugPoison(array, elems);
  }
  /** Bit-copy copy constructs an array of pointers. */
  template<typename T>
  inline T **CopyConstruct(T **dest, const T **src, size_t elems = 1) {
    return Copy(dest, src, elems);
  }

#undef BASE_CCMEM__SIMPLE_CONSTRUCTORS

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
  template<typename T>
  inline T *AllocCopyConstruct(const T *src, size_t elems = 1) {
    return AllocCopyConstruct<T, T>(src, elems);
  }
  /** Allocates and copy constructs an array. */
  template<typename T, typename U>
  inline T *AllocRepeatConstruct(const U &init, size_t elems) {
    return RepeatConstruct(Alloc<T>(elems), init, elems);
  }
  template<typename T>
  inline T *AllocRepeatConstruct(const T &init, size_t elems) {
    return AllocRepeatConstruct<T, T>(init, elems);
  }

  /** Destructs and frees an array. */
  template<typename T>
  inline void FreeDestruct(T *array, size_t elems = 1) {
    Free(Destruct(array, elems));
  }



  /** Offsets a pointer by a given number of bytes. */
  template<typename T>
  inline T *PtrAddBytes(T *ptr, ptrdiff_t bytes) {
    return reinterpret_cast<T *>(
        reinterpret_cast<char *>(ptr) + bytes);
  }
  /** Offsets a const pointer by a given number of bytes. */
  template<typename T>
  inline const T *PtrAddBytes(const T *ptr, ptrdiff_t bytes) {
    return reinterpret_cast<const T *>(
        reinterpret_cast<const char *>(ptr) + bytes);
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
    return reinterpret_cast<size_t>(lhs)
        == reinterpret_cast<size_t>(rhs);
  }

  ////////// Deprecated //////////////////////////////////////////////

  /** Renamed ZeroBytes */
  template<typename T>
  T *BitZeroBytes(T *array, size_t bytes) {
    return ZeroBytes(array, bytes);
  }
  /** Renamed Zero */
  template<typename T>
  T *BitZero(T *array, size_t elems = 1) {
    return Zero(array, elems);
  }
  /** Renamed CopyBytes */
  template<typename T, typename U>
  T *BitCopyBytes(T *dest, const U *src, size_t bytes) {
    return CopyBytes(dest, src, bytes);
  }
  /** Renamed Copy */
  template<typename T>
  T *BitCopy(T *dest, const T *src, size_t elems = 1) {
    return Copy(dest, src, elems);
  }
  /** Renamed Swap */
  template<typename T>
  void BitSwap(T *a, T *b, size_t elems = 1) {
    Swap(a, b, elems);
  }
};

#undef MEM__DEGUG_MEMORY

#endif
