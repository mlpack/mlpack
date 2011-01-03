/** @file parallel_sample_sort.h
 *
 *  An implementation of generic parallel sample sort.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_PARALLEL_SAMPLE_SORT_H
#define CORE_PARALLEL_PARALLEL_SAMPLE_SORT_H

#include <boost/mpi.hpp>

namespace core {
namespace parallel {
template<typename T>
class ParallelSampleSort {
  private:
    std::vector<T> *array_;

    int stride_;

    int num_samples_to_send_;

  private:
    void Sample_(std::vector<T> *sample_out) {
      int pos = 0;
      sample_out->resize(0);
      for(int i = 0; i < num_samples_to_send_;
          i++, pos = (pos + stride_) % array_->size()) {
        sample_out->push_back((*array_)[pos]);
      }
    }

  public:
    void Init(std::vector<T> &array_in, double sampling_rate_in) {
      array_ = &array_in;
      num_samples_to_send_ =
        static_cast<int>(ceil(sampling_rate_in * array_in.size()));
      stride_ = static_cast<int>(array_->size()) / num_samples_to_send_;
    }

    void Sort(boost::mpi::communicator &world) {

      // Locally sort the elements.
      std::sort(array_->begin(), array_->end());

      // Each process samples the dividers and sends to the master.
      std::vector<T> local_samples;
      Sample_(&local_samples);
      std::vector< std::vector<T> > collected_samples;
      boost::mpi::gather(world, local_samples, collected_samples, 0);

      // The master sorts the dividers and broadcasts $p$ partitions
      // to each process.

    }
};
};
};

#endif
