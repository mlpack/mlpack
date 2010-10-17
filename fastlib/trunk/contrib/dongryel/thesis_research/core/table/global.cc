/** @file global.cc
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include "core/table/memory_mapped_file.h"
#include "core/table/dense_point.h"

core::table::MemoryMappedFile *core::table::DensePoint::global_m_file_ = NULL;
