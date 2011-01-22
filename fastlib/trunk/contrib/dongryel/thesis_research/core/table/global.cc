/** @file global.cc
 *
 *  An instantiation of global memory mapped file object.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include "core/table/memory_mapped_file.h"

namespace core {
namespace table {

/** @brief A global memory mapped file object.
 */
core::table::MemoryMappedFile *global_m_file_ = NULL;
}
}
