/** @file parallel_sample_sort.h
 *
 *  An implementation of generic parallel sample sort.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_PARALLEL_SAMPLE_SORT_H
#define CORE_PARALLEL_PARALLEL_SAMPLE_SORT_H

namespace core {
namespace parallel {
template<typename T>
class ParallelSampleSort {
  private:
    std::vector<T> *array_;

  public:
    void Init(std::vector<T> &array_in) {
      array_ = &array_in;
    }

    void Sort() {
    }
};
};
};

#endif
