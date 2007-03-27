/**
 * @file fastalloc
 *
 * Lightweight specialized memory allocator for lots of equally-sized objects.
 */

#ifndef COL_FASTALLOC_H
#define COL_FASTALLOC_H

#include "base/common.h"
#include "base/ccmem.h"

/**
 * Fast memory allocator for identically sized chunks.
 *
 * This maintains a free list and never frees anything.
 */
template<size_t item_size>
class SlabAllocator {
 private:
  // TODO: what if we want to free the entire thing?
  static char *freelist_;
  //static char *slab_list_;
  
 public:
  static void AllocSlab();
  
  static char *Alloc() {
    if (unlikely(freelist_ == NULL)) {
      AllocSlab();
    }
    char *item = freelist_;
    freelist_ = *reinterpret_cast<char **>(freelist_);
    return item;
  }
  
  static void Free(char *item) {
    *reinterpret_cast<char**>(item) = freelist_;
    freelist_ = item;
  }
};

template<size_t item_size>
void SlabAllocator<item_size>::AllocSlab() {
  int slab_items = 128; // TODO: hard-coded constant
  size_t real_item_size = stride_align(item_size, char *);
  size_t size = real_item_size * slab_items + sizeof(char *);
  char *slab = mem::Alloc<char>(size);
  
  // slab_items must be at least 2
  
  *reinterpret_cast<char **>(slab) = freelist_;
  
  --slab_items;
  
  do {
    char *prev = slab;
    slab += real_item_size;
    *reinterpret_cast<char **>(slab) = prev;
  } while (--slab_items);
  
  freelist_ = slab;
  //slab += real_item_size;
  //
  //*reinterpret_cast<char **>(slab) = slab_list_;
  //slab_list_ = slab;
}

template<size_t item_size>
char *SlabAllocator<item_size>::freelist_ = 0;

/**
 * A very lightweight replacement for new for allocating lots of objects
 * of the same type.
 *
 * This allocator has zero space overhead and only a marginal per-allocation
 * overhead.  It is useful for cases where you need to allocate lots
 * of pointers to the same type of object and might need to free a lot
 * too.  It is not at all useful for arrays.
 *
 * Syntax is slightly different from new/delete:
 *
 * @code
 * MyClass *ptr1 = fast_new(MyClass);
 * MyClass *ptr2 = fast_new(MyClass)(constructor, parameters, go, here);
 * @endcode
 *
 * @param T the type being allocated
 */
#define fast_new(T) new(SlabAllocator<sizeof(T)>::Alloc()) T

/**
 * Delete operator for fast_new.
 *
 * @code
 * fast_delete(ptr);
 * @endcode
 *
 * @param ptr the pointer to destruct and free
 */
template<typename T>
inline void fast_delete(T *ptr) {
  ptr->~T();
  SlabAllocator<sizeof(T)>::Free(reinterpret_cast<char*>(ptr));
}

#endif
