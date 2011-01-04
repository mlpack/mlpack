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

    void Sample_(
      const std::vector<T> &array_in, int num_samples_in,
      std::vector<T> *sample_out) {
      int stride = static_cast<int>(array_in.size()) / num_samples_in;
      int pos = 0;
      sample_out->resize(0);
      for(int i = 0; i < num_samples_in;
          i++, pos = (pos + stride) % array_in.size()) {
        sample_out->push_back(array_in[pos]);
      }
    }

  public:
    void Init(std::vector<T> &array_in, double sampling_rate_in) {
      array_ = &array_in;
      num_samples_to_send_ =
        static_cast<int>(ceil(sampling_rate_in * array_in.size()));
    }

    template<typename PartitionFunctionType>
    void Sort(
      boost::mpi::communicator &world,
      const PartitionFunctionType &partition_function_in) {

      // Locally sort the elements.
      std::sort(array_->begin(), array_->end());

      // Each process samples the dividers and sends to the master.
      std::vector<T> local_samples;
      Sample_(*array_, num_samples_to_send_, &local_samples);
      std::vector< std::vector<T> > collected_samples;
      boost::mpi::gather(world, local_samples, collected_samples, 0);

      // The master sorts the dividers and broadcasts $p - 1$ partitions
      // to each process.
      std::vector<T> partitions;
      if(world.rank() == 0) {
        std::vector<T> flattened_samples;
        for(unsigned int i = 0; i < collected_samples.size(); i++) {
          flattened_samples.insert(
            flattened_samples.end(), collected_samples[i].begin(),
            collected_samples[i].end());
        }
        std::sort(flattened_samples.begin(), flattened_samples.end());
        Sample_(flattened_samples, world.size() - 1, &partitions);
      }
      boost::mpi::broadcast(world, partitions, 0);

      // Based on the $p$ partitions induced by the $p - 1$ numbers,
      // determine which part of the partitions each of the locally
      // owned points fall into, and do an all-to-all to make sure
      // that $i$-th process owns all points that fall into the $i$-th
      // partition.
      std::vector< std::vector<T> > local_buckets;
      std::vector< std::vector<T> > reshuffled_buckets;
      partition_function_in.Partition(*array_, partitions, &local_buckets);

      boost::mpi::all_to_all(world, local_buckets, reshuffled_buckets);
      // Flatten the reshuffled buckets.
      array_->resize(0);
      for(unsigned int i = 0; i < reshuffled_buckets.size(); i++) {
        array_->insert(
          array_->end(), reshuffled_buckets[i].begin(),
          reshuffled_buckets[i].end());
      }

      // Sort the reshuffled array.
      std::sort(array_->begin(), array_->end());
    }
};
};
};

#endif
