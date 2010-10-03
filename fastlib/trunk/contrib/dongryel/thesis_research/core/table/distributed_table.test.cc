/** @file distributed_table_test.cc
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include "boost/thread.hpp"
#include "core/table/distributed_table.h"

int main(int argc, char *argv[]) {
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;

  if(world.size() <= 1) {
    std::cout << "Please specify a process number greater than 1.\n";
    exit(0);
  }
  srand(time(NULL) + world.rank());
  printf("%d processes are present...\n", world.size());

  // Each process generates its own random data, dumps it to the file,
  // and read its own file back into its own distributed table.
  core::table::Table random_dataset;
  const int num_dimensions = 5;
  int num_points = core::math::RandInt(10000, 20000);
  random_dataset.Init(5, num_points);
  for(int j = 0; j < num_points; j++) {
    core::table::DensePoint point;
    random_dataset.get(j, &point);
    for(int i = 0; i < num_dimensions; i++) {
      point[i] = core::math::Random(0.1, 1.0);
    }
  }
  printf("Processor %d generated %d points...\n", world.rank(), num_points);
  std::stringstream file_name_sstr;
  file_name_sstr << "random_dataset_" << world.rank() << ".csv";
  std::string file_name = file_name_sstr.str();
  random_dataset.Save(file_name);

  core::table::DistributedTable distributed_table;
  distributed_table.Init(world.rank(), file_name, &world);
  printf(
    "Processor %d read in %d points...\n",
    world.rank(), distributed_table.local_n_entries());

  // Start the server for giving out points.
  boost::shared_ptr<boost::thread> point_server_thread(
    new boost::thread(
      boost::bind(
        &core::table::DistributedTable::remote_get,
        &distributed_table)));
  point_server_thread->detach();

  return 0;
}
