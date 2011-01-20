#ifndef GLOBAL_DATA_H
#define GLOBAL_DATA_H

#include "example.h"
#include "learner.h"

using namespace std;

typedef float T_LBL; // type for labels

struct global_data{
  bool quiet; //Should I keep quiet.

  T_IDX max_feature_idx; // maximum number of features in a dataset

  bool calc_loss; // calculate total loss.

  bool random_input; // randomly permute input examples.

  bool use_bias; // use a bias term (mainly for linear learners)

  size_t num_epoches; // number of training epoches.

  size_t num_iter_res; // number of training iterations besides epoches.

  string train_data_fn; // file name for the input data.

  string opt_method; // optimization method
  
  size_t num_threads; // number of threads.

  pthread_t par_read_thread; // for parallel data parsing

  int comm_method; // how agents communicate with each other

  int mb_size; // size of a mini-batch

  //Prediction output
  int final_prediction_sink; // set to send global predictions to.
  int raw_prediction; // file descriptors for text output.
  int local_prediction;  //file descriptor to send local prediction to.
  size_t unique_id; //unique id for each node in the network, id == 0 means extra io.

} global;

learner l1;

EXAMPLE *train_exps; // training examples.
size_t num_train_exps; // total # of training examples

size_t *old_from_new;

size_t epoch_ct; // counter of epoches used.
size_t iter_res_ct; // counter of residual number of iterations.
size_t parsed_ct; // How many examples been parsed.
size_t used_ct; // How many examples used, Can be larger than parsed_index.
//size_t *thread_using_index; // The index of the example currently using by thread i. starts from 0.

size_t ring_size;
//size_t *delay_indicies;
//size_t *threads_to_use;
//EXAMPLE **delay_ring;

#endif
