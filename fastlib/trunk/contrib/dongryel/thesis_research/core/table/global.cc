/** @file global.cc
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include "memory_mapped_file.h"
#include "dense_matrix.h"
#include "dense_point.h"
#include "table.h"

namespace core {
namespace table {
core::table::MemoryMappedFile *global_m_file_ = NULL;
};
};
