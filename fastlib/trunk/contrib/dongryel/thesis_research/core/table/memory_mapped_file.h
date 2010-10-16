/** @file memory_mapped_file.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_MEMORY_MAPPED_FILE_H
#define CORE_TABLE_MEMORY_MAPPED_FILE_H

#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/managed_mapped_file.hpp>

namespace core {
namespace table {
class MemoryMappedFile {

  private:
    boost::interprocess::managed_mapped_file m_file_;

    boost::interprocess::interprocess_mutex *mutex_;

  public:

    MemoryMappedFile(): m_file_(
        boost::interprocess::open_or_create, "tmp_file", 500000000) {
      mutex_ = (boost::interprocess::interprocess_mutex *)
               m_file_.allocate(
                 sizeof(boost::interprocess::interprocess_mutex));
    }

    void Init(const std::string &file_name) {
      boost::interprocess::managed_mapped_file new_m_file(
        boost::interprocess::create_only, file_name.c_str(), 500000000);
      m_file_.swap(new_m_file);
    }

    void *Allocate(size_t size) {
      boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> lock(*mutex_);
      return m_file_.allocate(size);
    }

    void Deallocate(void *p) {
      boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> lock(*mutex_);
      return m_file_.deallocate(p);
    }
};
};
};

#endif
