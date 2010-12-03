/** @file distributed_kde.test.cc
 *
 *  The test driver for the distributed kde.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include "core/metric_kernels/lmetric.h"
#include "core/tree/statistic.h"
#include "core/table/distributed_table.h"
#include "core/table/mailbox.h"
#include "core/tree/gen_kdtree.h"
#include "core/tree/gen_metric_tree.h"
#include "mlpack/kde/kde_dualtree.h"

typedef core::tree::GenMetricTree<core::tree::AbstractStatistic> TreeSpecType;
typedef core::tree::GeneralBinarySpaceTree < TreeSpecType > TreeType;
typedef core::table::Table<TreeSpecType> TableType;

void Compute(
  core::table::DistributedTable<TreeSpecType> *distributed_table_in) {

  // Each process does the work owned by itself.

  // This is the exchange loop.
  do {

    // Each process grabs the necessary work. This is the exchange
    // phase.

    // Each process computes.

  }
  while(true);
}

core::table::DistributedTable<TreeSpecType> *InitDistributedTable(
  boost::mpi::communicator &world) {

  core::table::DistributedTable<TreeSpecType> *distributed_table =
    NULL;

  printf("Process %d: TableOutbox.\n", world.rank());

  // Each process generates its own random data, dumps it to the file,
  // and read its own file back into its own distributed table.
  core::table::Table<TreeSpecType> random_dataset;
  const int num_dimensions = 5;
  int num_points = core::math::RandInt(10, 20);
  random_dataset.Init(5, num_points);
  for(int j = 0; j < num_points; j++) {
    core::table::DensePoint point;
    random_dataset.get(j, &point);
    for(int i = 0; i < num_dimensions; i++) {
      point[i] = core::math::Random(0.1, 1.0);
    }
  }
  printf("Process %d generated %d points...\n", world.rank(), num_points);
  std::stringstream file_name_sstr;
  file_name_sstr << "random_dataset_" << world.rank() << ".csv";
  std::string file_name = file_name_sstr.str();
  random_dataset.Save(file_name);

  std::stringstream distributed_table_name_sstr;
  distributed_table_name_sstr << "distributed_table_" << world.rank() << "\n";
  distributed_table = core::table::global_m_file_->UniqueConstruct <
                      core::table::DistributedTable<TreeSpecType> > ();
  distributed_table->Init(file_name, world);
  printf(
    "Process %d read in %d points...\n",
    world.rank(), distributed_table->local_n_entries());
  return distributed_table;
}

int main(int argc, char *argv[]) {

  // Initialize boost MPI.
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;

  // Delete the teporary files and put a barrier.
  std::stringstream temporary_file_name;
  temporary_file_name << "tmp_file" << world.rank();
  remove(temporary_file_name.str().c_str());
  world.barrier();

  // Initialize the memory allocator.
  core::table::global_m_file_ = new core::table::MemoryMappedFile();
  core::table::global_m_file_->Init(
    std::string("tmp_file"), world.rank(), world.rank(), 50000000);

  // Seed the random number.
  srand(time(NULL) + world.rank());

  if(world.rank() == 0) {
    printf("%d processes are present...\n", world.size());
  }

  // Read in the distributed table, and index the global tree.
  core::table::DistributedTable<TreeSpecType> *distributed_table =
    InitDistributedTable(world);
  core::metric_kernels::LMetric<2> l2_metric;
  distributed_table->IndexData(l2_metric, world, 3, 0.5);

  // Enter the main computation loop.
  Compute(distributed_table);

  // Free the intercommunicators.
  world.barrier();

  return 0;
}
