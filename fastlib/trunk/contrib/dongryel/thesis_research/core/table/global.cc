/** @file global.cc
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include "memory_mapped_file.h"
#include "dense_matrix.h"
#include "dense_point.h"
#include "table.h"

core::table::MemoryMappedFile *core::table::DenseMatrix::global_m_file_ = NULL;

core::table::MemoryMappedFile *core::table::DensePoint::global_m_file_ = NULL;

core::table::MemoryMappedFile *core::table::Table::global_m_file_ = NULL;
