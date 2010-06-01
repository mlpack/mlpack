#include "memory_manager.h"

namespace mmapmm {

  /**
   * These global variables are necessary to instanciate a memory manager
   */
  template<>
  MemoryManager<false>  *MemoryManager<false>::allocator_ = 0;
  template<>
  MemoryManager<true>  *MemoryManager<true>::allocator_ = 0;
};
