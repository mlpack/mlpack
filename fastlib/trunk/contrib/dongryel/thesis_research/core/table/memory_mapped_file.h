/** @file memory_mapped_file.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_MEMORY_MAPPED_FILE_H
#define CORE_TABLE_MEMORY_MAPPED_FILE_H

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/managed_mapped_file.hpp>
#include <iostream>
#include <sstream>

namespace core {
namespace table {
class MemoryMappedFile {

  private:
    boost::interprocess::managed_mapped_file m_file_;

    std::string m_file_name_;

  public:

    ~MemoryMappedFile() {
      boost::interprocess::shared_memory_object::remove(m_file_name_.c_str());
    }

    boost::interprocess::managed_mapped_file &m_file() {
      return m_file_;
    }

    void Init(
      const std::string &file_name,
      int world_rank, int group_rank, long int num_bytes) {

      // Get the file name.
      m_file_name_ = file_name;

      std::stringstream new_file_name;
      new_file_name << file_name << group_rank;

      boost::interprocess::managed_mapped_file new_m_file(
        boost::interprocess::open_or_create,
        new_file_name.str().c_str(), num_bytes);
      m_file_.swap(new_m_file);
      m_file_.zero_free_memory();

      std::cout << "World rank " << world_rank <<
                " opened the memory mapped file: " <<
                new_file_name.str() << "\n";
    }

    template<typename MyType>
    std::pair<MyType *, std::size_t> UniqueFind() {
      return m_file_.find<MyType>(boost::interprocess::unique_instance);
    }

    template<typename MyType>
    MyType *UniqueConstruct() {
      return m_file_.construct<MyType>(boost::interprocess::unique_instance)();
    }

    template<typename MyType>
    MyType *Construct() {
      return m_file_.construct<MyType>(
               boost::interprocess::anonymous_instance)();
    }

    void *Allocate(size_t size) {
      return m_file_.allocate(size);
    }

    void Deallocate(void *p) {
      m_file_.deallocate(p);
    }

    template<typename MyType>
    void DestroyPtr(MyType *ptr) {
      m_file_.destroy_ptr(ptr);
    }

    template<typename ReturnType>
    ReturnType *ConstructArray(int num_elements_in) {
      return m_file_.construct<ReturnType>(
               boost::interprocess::anonymous_instance)[num_elements_in]();
    }
};
};
};

#endif
