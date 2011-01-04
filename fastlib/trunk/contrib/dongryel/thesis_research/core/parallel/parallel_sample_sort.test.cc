/** @file parallel_sample_sort.test.cc
 *
 *  A simple test for distributed parallel sample sort algorithm.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include "core/parallel/parallel_sample_sort.h"
#include "core/math/math_lib.h"

class DoublePartitionFunction {
  private:
    int ComputeBucketIndex_(
      const std::vector<double> &partitions_in, double element_in) const {

      int bucket_index = 0;
      for(unsigned int i = 0; i < partitions_in.size(); i++) {
        if(element_in > partitions_in[i]) {
          bucket_index++;
        }
        else {
          break;
        }
      }
      return bucket_index;
    }

  public:
    void Partition(
      const std::vector<double> &array_in,
      const std::vector<double> &partitions_in,
      std::vector< std::vector<double> > *buckets_out) const {

      buckets_out->resize(0);
      buckets_out->resize(partitions_in.size() + 1);
      for(unsigned int i = 0; i < array_in.size(); i++) {
        int bucket_index = ComputeBucketIndex_(partitions_in, array_in[i]);
        (*buckets_out)[bucket_index].push_back(array_in[i]);
      }
    }
};

int main(int argc, char *argv[]) {

  // Initialize boost MPI.
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;

  // Seed the random number generator.
  core::math::global_random_number_state_.set_seed(time(NULL) + world.rank());

  // Create a random vector of weights.
  std::vector<double> weights(world.size(), 0);
  for(unsigned int i = 0; i < weights.size(); i++) {
    weights[i] = core::math::Random(0.5, 5.0);
    printf("Weight %d: %g\n", world.rank(), weights[i]);
  }

  // Sort.
  core::parallel::ParallelSampleSort<double> sorter;
  DoublePartitionFunction partition_function;
  sorter.Init(weights, 0.2);
  sorter.Sort(world, partition_function);
  for(unsigned int i = 0; i < weights.size(); i++) {
    printf("Sorted weight %d: %g\n", world.rank(), weights[i]);
  }

  return 0;
}
