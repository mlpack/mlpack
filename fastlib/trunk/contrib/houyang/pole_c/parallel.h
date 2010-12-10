#ifndef PARALLEL_H
#define PARALLEL_H


#include <pthread.h>

// parameters of a single thread
struct thread_param {
  size_t thread_id;
  learner *l;
  size_t thread_state;
  // 0: waiting to read data
  // 1: data read, predict and send message(e.g. calc subgradient)
  // 2: msg sent done, waiting to receive messages from other agents and update
};

pthread_t* threads;
thread_param** t_par;

pthread_mutex_t examples_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t example_delay = PTHREAD_MUTEX_INITIALIZER;

pthread_cond_t example_available = PTHREAD_COND_INITIALIZER;

pthread_barrier_t barrier_msg_all_sent;
pthread_barrier_t barrier_msg_all_used;

void FinishThreads(size_t n_threads) {
  for (size_t i=0; i<n_threads; i++) {
    pthread_join(threads[i], NULL);
    free(t_par[i]);
  }
  free(threads);
  free(t_par);
}


#endif
