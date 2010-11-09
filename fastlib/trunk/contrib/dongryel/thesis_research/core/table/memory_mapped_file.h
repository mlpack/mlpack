/** @file memory_mapped_file.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_MEMORY_MAPPED_FILE_H
#define CORE_TABLE_MEMORY_MAPPED_FILE_H

#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/managed_mapped_file.hpp>
#include <iostream>
#include <sstream>

namespace core {
namespace table {
class MemoryMappedFile {

  private:
    boost::interprocess::managed_mapped_file m_file_;

    boost::interprocess::interprocess_mutex *mutex_;

  public:

    void Init(
      const std::string &file_name,
      int world_rank, int group_rank, long int num_bytes) {

      std::stringstream new_file_name;
      new_file_name << file_name << group_rank;
      std::stringstream new_mutex_name;
      new_mutex_name << "mtx" << group_rank;
      boost::interprocess::managed_mapped_file new_m_file(
        boost::interprocess::open_or_create,
        new_file_name.str().c_str(), num_bytes);
      m_file_.swap(new_m_file);
      mutex_ = m_file_.find_or_construct <
               boost::interprocess::interprocess_mutex > (
                 new_mutex_name.str().c_str())();
      std::cout << "World rank " << world_rank <<
                " opened the memory mapped file: " <<
                new_file_name.str() << "\n";
      std::cout << "World rank " << world_rank <<
                " opened the mutex on the memory mapped file: "
                << new_mutex_name.str() << "\n";
    }

    template<typename MyType>
    std::pair<MyType *, std::size_t> UniqueFind() {
      boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> lock(*mutex_);
      return m_file_.find<MyType>(boost::interprocess::unique_instance);
    }

    template<typename MyType>
    MyType *UniqueConstruct() {
      boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> lock(*mutex_);
      return m_file_.construct<MyType>(boost::interprocess::unique_instance)();
    }

    void *Allocate(size_t size) {
      boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> lock(*mutex_);
      return m_file_.allocate(size);
    }

    void Deallocate(void *p) {
      boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> lock(*mutex_);
      m_file_.deallocate(p);
    }

    template<typename MyType>
    void DestroyPtr(MyType *ptr) {
      boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> lock(*mutex_);
      m_file_.destroy_ptr(ptr);
    }
};
};
};

#endif
