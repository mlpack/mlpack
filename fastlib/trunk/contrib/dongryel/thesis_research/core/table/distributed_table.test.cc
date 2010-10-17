/** @file distributed_table_test.cc
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */
#include "core/metric_kernels/lmetric.h"
#include "core/table/distributed_table.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <new>

core::table::MemoryMappedFile *core::table::DensePoint::global_m_file_ = NULL;

bool CheckDistributedTableIntegrity(
  const core::table::DistributedTable &table_in,
  const boost::mpi::communicator &world) {
  for(int i = 0; i < world.size(); i++) {
    printf("Process %d thinks Process %d owns %d points.\n",
           world.rank(), i, table_in.local_n_entries(i));
  }
  return true;
}

/*
void TestDistributedTree(boost::mpi::communicator &world) {

  printf("------------------------------------\n");
  printf("Process %d in TestDistributedTree...\n", world.rank());
  printf("------------------------------------\n");

  // Each process generates its own random data, dumps it to the file,
  // and read its own file back into its own distributed table.
  core::table::Table random_dataset;
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

  core::table::DistributedTable distributed_table;
  distributed_table.Init(file_name, &world);
  printf(
    "Process %d read in %d points...\n",
    world.rank(), distributed_table.local_n_entries());

  printf("Building the distributed tree...\n");
  core::metric_kernels::LMetric<2> l2_metric;
  distributed_table.IndexData(l2_metric, 0.3);

  // Print out the tree.
  if(world.rank() == world.size() / 2) {
    printf("Processor %d prints out the tree.\n", world.rank());
    distributed_table.get_tree()->Print();
  }

  // Put a barrier.
  world.barrier();
}

void TestDistributedTable(boost::mpi::communicator &world) {

  printf("------------------------------------\n");
  printf("Process %d in TestDistributedTable..\n", world.rank());
  printf("------------------------------------\n");

  // Each process generates its own random data, dumps it to the file,
  // and read its own file back into its own distributed table.
  core::table::Table random_dataset;
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

  core::table::DistributedTable distributed_table;
  distributed_table.Init(file_name, &world);
  printf(
    "Process %d read in %d points...\n",
    world.rank(), distributed_table.local_n_entries());

  // Check the integrity.
  CheckDistributedTableIntegrity(distributed_table, world);

  // Each process exchanges 1000 points.
  for(int i = 0; i < 12; i++) {

    // Randomly generate a target.
    int target_rank = core::math::RandInt(world.size());
    core::table::DensePoint point;
    int target_rank_table_n_entries = distributed_table.local_n_entries(
                                        target_rank);
    int target_point_id = core::math::RandInt(target_rank_table_n_entries);

    printf("Process %d requested point %d from Process %d.\n",
           world.rank(), target_point_id, target_rank);
    distributed_table.get(target_rank, target_point_id, &point);

    printf("Process %d received point %d of length %d from Process %d.\n",
           world.rank(), target_point_id, point.reference().n_elem, target_rank);
    point.reference().print();
  }
  printf("Process %d is all done!\n", world.rank());

  // Put a barrier.
  world.barrier();
}
*/

void PointRequestMessageBoxProcess(
  boost::mpi::communicator &world,
  boost::mpi::communicator &table_group) {
  printf("Process %d: PointRequestMessageBox.\n", world.rank());

  // Each process generates its own random data, dumps it to the file,
  // and read its own file back into its own distributed table.
  core::table::Table random_dataset;
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

  core::table::DistributedTable distributed_table;
  core::table::DensePoint::global_m_file_ = &distributed_table.global_m_file();
  distributed_table.Init(file_name, &world, &table_group);
  printf(
    "Process %d read in %d points...\n",
    world.rank(), distributed_table.local_n_entries());

}

void PointInboxProcess(boost::mpi::communicator &world) {
  printf("Process %d: PointInbox.\n", world.rank());

}

void ComputationProcess(boost::mpi::communicator &world) {
  printf("Process %d: Computation.\n", world.rank());

}

int main(int argc, char *argv[]) {

  // Initialize boost MPI.
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;

  if(world.size() <= 3 || world.size() % 3 != 0) {
    std::cout << "Please specify a process number greater than 1 and "
              "a multiple of 3.\n";
    return 0;
  }
  srand(time(NULL) + world.rank());

  if(world.rank() == 0) {
    printf("%d processes are present...\n", world.size());
  }

  // If the process ID is less than half of the size of the
  // communicator, make it a table process. Otherwise, make it a
  // computation process. This assignment depends heavily on the
  // round-robin assignment of mpirun.
  boost::mpi::communicator table_group = world.split(
      (world.rank() < world.size() / 3) ? 1 : 0);
  if(world.rank() < world.size() / 3) {
    PointRequestMessageBoxProcess(world, table_group);
  }
  else if(world.rank() < world.size() / 3 * 2) {
    PointInboxProcess(world);
  }
  else {
    ComputationProcess(world);
  }

  // Test the distributed table point exchanges.
  //TestDistributedTable(world);

  // Test the distributed tree building.
  //TestDistributedTree(world);
  return 0;
}
