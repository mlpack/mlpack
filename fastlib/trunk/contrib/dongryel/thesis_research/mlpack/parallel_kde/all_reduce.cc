/** @file all_reduce.cc
 *
 *  @author Dongryeol Lee (gth840e@mail.gatech.edu)
 *
 *  An implementation of the all-reduce operation using MPI send and recv's.
 */

#include <unistd.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <time.h>
#include <values.h>
#include <stdlib.h>

int RoundDownToNearestPowerofTwo(int num) {
  unsigned int v = static_cast<unsigned int>(num);
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v++;
  if(v > static_cast<unsigned int>(num)) {
    v = v >> 1;
  }
  return static_cast<int>(v);
}

double ApplyOp(MPI_Op op, double first_arg, double second_arg) {
  double result = 0;
  if(op == MPI_SUM) {
    result = first_arg + second_arg;
  }
  else if(op == MPI_MAX) {
    result = (first_arg > second_arg) ? first_arg : second_arg;
  }
  else if(op == MPI_MIN) {
    result = (first_arg < second_arg) ? first_arg : second_arg;
  }
  return result;
}

int Allreduce(void *sbuf, void *rbuf, int count, MPI_Op op, MPI_Comm comm) {

  // write your own code using MPI point to point ops to implement the
  // CS6230_Allreduce here...
  //
  // the equivalent call using the MPI collective is
  //
  // MPI_Allreduce (sbuf, rbuf, count, MPI_DOUBLE, op, comm);

  // The rank of the process.
  int rank;
  int num_processes;
  MPI_Comm_size(comm, &num_processes);
  MPI_Comm_rank(comm, &rank);

  // Initialize the starting value for the sub-reduction result for
  // the data owned by the current process.
  double *sbuf_double = (double *) sbuf;
  double sub_reduction;
  if(op == MPI_SUM) {
    sub_reduction = 0;
  }
  else if(op == MPI_MAX) {
    sub_reduction = -DBL_MAX;
  }
  else if(op == MPI_MIN) {
    sub_reduction = DBL_MAX;
  }
  else {
    return MPI_ERR_OP;
  }

  // Reduce over the data held by the current process.
  if(count > 0) {
    sub_reduction = sbuf_double[0];
    int i;

    // Pair up for efficiency.
    for(i = 1; i < count - 1; i += 2) {
      double pair_reduction = ApplyOp(op, sbuf_double[i], sbuf_double[i + 1]);
      sub_reduction = ApplyOp(op, sub_reduction, pair_reduction);
    }
    if(count % 2 == 0) {
      sub_reduction = ApplyOp(op, sub_reduction, sbuf_double[count - 1]);
    }
  }

  // Do the first half of the prefix scan operation so that the last
  // process holds the result.

  // This step is necessary to take care of the case when the number
  // of proceses is not exactly a power of two.
  int rounded_down_num_processes = RoundDownToNearestPowerofTwo(num_processes);
  int i;
  int stride = num_processes - rounded_down_num_processes;
  if(rank >= rounded_down_num_processes && rank < num_processes) {

    // Each extra process sends to its partner.
    MPI_Send(&sub_reduction, 1, MPI_DOUBLE, rank - stride, rank, comm);
  }
  if(rank >= rounded_down_num_processes - stride &&
      rank < num_processes - stride) {

    // Receive from its partnering extra process.
    double received;
    MPI_Request receive_request;
    MPI_Status status;
    MPI_Irecv(
      &received, 1, MPI_DOUBLE, rank + stride, MPI_ANY_TAG,
      comm, &receive_request);
    MPI_Wait(&receive_request, &status);
    sub_reduction = ApplyOp(op, sub_reduction, received);
  }

  // This is the perfectly balanced binary tree for the remaining
  // processes.
  if(rank < rounded_down_num_processes) {
    int sub_stride;
    int starting_index = 0;
    for(sub_stride = 2; sub_stride < rounded_down_num_processes * 2;
        sub_stride = sub_stride << 1) {
      for(i = starting_index; i < rounded_down_num_processes; i += sub_stride) {

        // If the current process ID's bit mask does not match i, then send.
        if(i == rank) {
          MPI_Send(
            &sub_reduction, 1, MPI_DOUBLE, rank + sub_stride / 2, rank, comm);
          break;
        }

        // If it does, then receive from its partner.
        else if(rank == i + sub_stride / 2) {
          double received;
          MPI_Request receive_request;
          MPI_Status status;
          MPI_Irecv(
            &received, 1, MPI_DOUBLE, rank - sub_stride / 2, MPI_ANY_TAG,
            comm, &receive_request);
          MPI_Wait(&receive_request, &status);
          sub_reduction = ApplyOp(op, sub_reduction, received);
          break;
        }
      }

      // Modify the starting index for the striding.
      starting_index += (sub_stride >> 1);
    }

    // The process ID equal to (rounded_down_num_processes - 1) holds
    // the final results, which should be broadcasted using a balanced
    // binary tree rooted at it.
    for(sub_stride = rounded_down_num_processes; sub_stride >= 2;
        sub_stride = sub_stride >> 1) {
      for(i = rounded_down_num_processes - 1; i >= 0; i -= sub_stride) {

        // The i-th process sends to (i - sub_stride / 2) th process.
        if(rank == i) {
          MPI_Send(
            &sub_reduction, 1, MPI_DOUBLE, rank - sub_stride / 2, rank, comm);
          break;
        }
        else if(rank == i - sub_stride / 2) {
          double received;
          MPI_Request receive_request;
          MPI_Status status;
          MPI_Irecv(
            &received, 1, MPI_DOUBLE, rank + sub_stride / 2, MPI_ANY_TAG,
            comm, &receive_request);
          MPI_Wait(&receive_request, &status);
          sub_reduction = received;
          break;
        }
      }
    }
  }

  // In case the extra processes need the reduction result,
  if(stride > 0) {
    if(rank >= rounded_down_num_processes - stride &&
        rank < num_processes - stride) {

      // Send to the partner.
      MPI_Send(&sub_reduction, 1, MPI_DOUBLE, rank + stride, rank, comm);
    }
    if(rank >= rounded_down_num_processes && rank < num_processes) {

      // Receive from the partner.
      double received;
      MPI_Request receive_request;
      MPI_Status status;
      MPI_Irecv(
        &received, 1, MPI_DOUBLE, rank - stride, MPI_ANY_TAG,
        comm, &receive_request);
      MPI_Wait(&receive_request, &status);
      sub_reduction = received;
    }
  }

  // Write the result out to the output.
  ((double *)rbuf)[0] = sub_reduction;

  return MPI_SUCCESS;
}

int main(int ac, char *av[]) {
  int i;
  int rank = -1;
  int size = -1;
  int count = rand() % 10;
  const int maxsize = 256 * 256;
  double start, stop;
  double sbuf[maxsize];
  double rbuf[maxsize];

  MPI_Init(&ac, &av);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  srand(time(NULL) + rank);
  for(i = 0; i < count; i++) {
    sbuf[i] = rand() % 10;
  }

  // Find out the nearest power of two that is smaller and closest to
  // the size of the MPI communicator.
  if(rank == 0) {
    printf(
      "The world size is %d, rounded down to nearest power of two is %d.\n",
      size, RoundDownToNearestPowerofTwo(size));
  }

  start = MPI_Wtime();
  Allreduce(sbuf, rbuf, count, MPI_SUM, MPI_COMM_WORLD);
  stop = MPI_Wtime();

  if(rank == 0) {
    printf("Runtime = %g\n"
           "Calls = %d\n"
           "Time per call = %g us\n",
           stop - start, count, 1e6 *(stop - start) / count);
  }

  MPI_Finalize();
  return 0;
}
