/** @file distributed_kde.test.cc
 *
 *  The test driver for the distributed kde.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include "core/metric_kernels/lmetric.h"
#include "core/table/distributed_table.h"
#include "core/table/mailbox.h"
#include "core/tree/gen_kdtree.h"
#include "core/tree/gen_metric_tree.h"
#include "mlpack/kde/kde_dualtree.h"

typedef core::tree::GenMetricTree<mlpack::kde::KdeStatistic> TreeSpecType;
typedef core::tree::GeneralBinarySpaceTree < TreeSpecType > TreeType;
typedef core::table::Table<TreeType> TableType;

core::table::DistributedTable<TreeSpecType> *InitDistributedTable(
  boost::mpi::communicator &world,
  boost::mpi::communicator &table_outbox_group) {

  std::pair< core::table::DistributedTable<TreeSpecType> *, std::size_t >
  distributed_table_pair =
    core::table::global_m_file_->UniqueFind <
    core::table::DistributedTable<TreeSpecType> > ();
  core::table::DistributedTable<TreeSpecType> *distributed_table =
    distributed_table_pair.first;

  if(distributed_table == NULL) {
    printf("Process %d: TableOutbox.\n", world.rank());

    // Each process generates its own random data, dumps it to the file,
    // and read its own file back into its own distributed table.
    core::table::Table<TreeType> random_dataset;
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
    file_name_sstr << "random_dataset_" << table_outbox_group.rank() << ".csv";
    std::string file_name = file_name_sstr.str();
    random_dataset.Save(file_name);

    std::stringstream distributed_table_name_sstr;
    distributed_table_name_sstr << "distributed_table_" << world.rank() << "\n";
    distributed_table = core::table::global_m_file_->UniqueConstruct <
                        core::table::DistributedTable<TreeSpecType> > ();
    distributed_table->Init(
      file_name, table_outbox_group);
    printf(
      "Process %d read in %d points...\n",
      world.rank(), distributed_table->local_n_entries());
  }
  return distributed_table;
}

void TableOutboxProcess(
  core::table::DistributedTable<TreeSpecType> *distributed_table,
  boost::mpi::communicator &world,
  boost::mpi::intercommunicator &outbox_to_inbox_comm,
  boost::mpi::intercommunicator &outbox_to_computation_comm) {

  printf("Process %d: TableOutbox.\n", world.rank());
  distributed_table->RunOutbox(
    outbox_to_inbox_comm, outbox_to_computation_comm);
}

void TableInboxProcess(
  core::table::DistributedTable<TreeSpecType> *distributed_table,
  boost::mpi::communicator &world,
  boost::mpi::intercommunicator &inbox_to_outbox_comm,
  boost::mpi::intercommunicator &inbox_to_computation_comm) {
  printf("Process %d: TableInbox.\n", world.rank());

  distributed_table->RunInbox(
    inbox_to_outbox_comm, inbox_to_computation_comm);
}

void ComputationProcess(
  core::table::DistributedTable<TreeSpecType> *distributed_table,
  boost::mpi::communicator &world,
  boost::mpi::communicator &local_group_comm,
  boost::mpi::intercommunicator &computation_to_outbox_comm,
  boost::mpi::intercommunicator &computation_to_inbox_comm) {

  printf("Process %d: Computation.\n", world.rank());

  // Do a test where each computation process requests a random point
  // from a randomly chosen process.
  int num_points = core::math::RandInt(10, 30);
  for(int n = 0; n < num_points; n++) {
    core::table::DensePoint point;
    int random_request_rank = core::math::RandInt(
                                0, computation_to_outbox_comm.remote_size());
    int random_request_point_id =
      core::math::RandInt(
        0, distributed_table->local_n_entries(random_request_rank));
    printf("Computation Process %d is requesting point %d from Table Outbox "
           "Process %d\n",
           local_group_comm.rank(), random_request_point_id,
           random_request_rank);
    distributed_table->get(
      computation_to_outbox_comm, computation_to_inbox_comm,
      random_request_rank, random_request_point_id, &point);

    // Print the point.
    point.Print();

    // Tell the inbox that we are done using the point.
    distributed_table->UnlockPointinTableInbox();
  }

  // Barrier so that all computation groups are here, at which outbox
  // and inboxes are terminated.
  printf("Notifying all mailboxes that Computation group %d is done!\n",
         local_group_comm.rank());
  for(int i = 0; i < computation_to_outbox_comm.remote_size(); i++) {
    computation_to_outbox_comm.isend(
      i, core::table::DistributedTableMessage::TERMINATE_TABLE_OUTBOX,
      0);
    computation_to_inbox_comm.isend(
      i, core::table::DistributedTableMessage::TERMINATE_TABLE_INBOX,
      0);
  }
}

int main(int argc, char *argv[]) {

  // Initialize boost MPI.
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;

  if(world.size() <= 1 || world.size() % 3 != 0) {
    std::cout << "Please specify a process number greater than 1 and "
              "a multiple of 3.\n";
    return 0;
  }

  // Delete the teporary files and put a barrier.
  std::stringstream temporary_file_name;
  temporary_file_name << "tmp_file" << world.rank();
  remove(temporary_file_name.str().c_str());
  world.barrier();

  // Initialize the memory allocator.
  int membership_key = world.rank() % 3;
  core::table::global_m_file_ = new core::table::MemoryMappedFile();
  core::table::global_m_file_->Init(
    std::string("tmp_file"), world.rank(),
    (int) floor(world.rank() / 3), 5000000);

  // Seed the random number.
  srand(time(NULL) + world.rank());

  if(world.rank() == 0) {
    printf("%d processes are present...\n", world.size());
  }

  // Split the world communicator into three groups: the first group
  // that sends stuffs to other processes, the second group that
  // receives stuffs from other processes, and the third group that
  // does the computation.
  boost::mpi::communicator local_group_comm = world.split(membership_key);

  // Build the intercommunicator between the table outbox group and
  // the table inbox group and the computation group.
  boost::mpi::intercommunicator *first_inter_comm = NULL;
  boost::mpi::intercommunicator *second_inter_comm = NULL;
  if(membership_key == 0) {
    first_inter_comm = new boost::mpi::intercommunicator(
      local_group_comm, 0, world, 1);
    second_inter_comm = new boost::mpi::intercommunicator(
      local_group_comm, 0, world, 2);
  }
  else if(membership_key == 1) {
    first_inter_comm = new boost::mpi::intercommunicator(
      local_group_comm, 0, world, 0);
    second_inter_comm = new boost::mpi::intercommunicator(
      local_group_comm, 0, world, 2);
  }
  else {
    first_inter_comm = new boost::mpi::intercommunicator(
      local_group_comm, 0, world, 0);
    second_inter_comm = new boost::mpi::intercommunicator(
      local_group_comm, 0, world, 1);
  }

  // Declare the distributed table.
  core::table::DistributedTable<TreeSpecType> *distributed_table = NULL;

  // Wait until the memory allocator is in synch.
  world.barrier();

  // Read the distributed table once per each compute node, and put a
  // barrier.
  if(membership_key == 0) {
    distributed_table =
      InitDistributedTable(world, local_group_comm);
  }
  world.barrier();

  // Attach the distributed table for all the processes and put a
  // barrier.
  std::pair< core::table::DistributedTable<TreeSpecType> *, std::size_t >
  distributed_table_pair =
    core::table::global_m_file_->UniqueFind <
    core::table::DistributedTable<TreeSpecType> > ();
  distributed_table = distributed_table_pair.first;

  // The main computation loop.
  if(membership_key == 0) {
    TableOutboxProcess(
      distributed_table, world, *first_inter_comm, *second_inter_comm);
  }
  else if(membership_key == 1) {
    TableInboxProcess(
      distributed_table, world, *first_inter_comm, *second_inter_comm);
  }
  else {
    ComputationProcess(
      distributed_table, world, local_group_comm,
      *first_inter_comm, *second_inter_comm);
  }

  // Free the intercommunicators.
  world.barrier();
  delete first_inter_comm;
  delete second_inter_comm;

  return 0;
}
