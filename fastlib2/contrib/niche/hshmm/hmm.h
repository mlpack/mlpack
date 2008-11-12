#ifndef HMM_H
#define HMM_H


#include "fastlib/fastlib.h"

#include "distribution.h"

class HMM {

 private:

  int n_states_;
  int n_dims_;

  Vector p_initial_;

  Matrix p_transition_;

  ArrayList<Distribution> state_distributions_;

    

 public:

  void Init(int n_states_in, int n_dims_in) {
    n_states_ = n_states_in;
    n_dims_ = n_dims_in;
    
    p_initial_.Init(n_states_);
    p_transition_.Init(n_states_, n_states_);
    state_distributions_.Init(n_states_);
    for(int i = 0; i < n_states_; i++) {
      state_distributions_[i].Init(n_dims_in);
    }
  }

  void RandomlyInitialize() {
    double uniform = ((double) 1) / ((double) n_states_);

    for(int i = 0; i < n_states_; i++) {
      p_initial_[i] = uniform;
      
      for(int j = 0; j < n_states_; j++) {
	p_transition_.set(j, i, uniform);
      }

      state_distributions_[i].RandomlyInitialize();
    }
  }

  int n_states() {
    return n_states_;
  }

  int n_dims() {
    return n_dims_;
  }
  
  Vector p_initial() {
    return p_initial_;
  }

  Matrix p_transition() {
    return p_transition_;
  }

  ArrayList<Distribution> state_distributions() {
    return state_distributions_;
  }

  
  void PrintDebug(const char *name = "", FILE *stream = stderr) /*const*/ {
    fprintf(stream, "----- HMM %s ------\n", name);
    
    p_initial_.PrintDebug("initial probabilities", stream);
    p_transition_.PrintDebug("transition probabilities", stream);
    
    for(int i = 0 ;i < n_states_; i++) {
      fprintf(stream, "state %d:\n", i+1);
      state_distributions_[i].mu().PrintDebug("mu");
      state_distributions_[i].sigma().PrintDebug("sigma");
      fprintf(stream, "\n");
    }
  }
  
  /* this function calculates P(q_t = s_i | theta) */
  void CalculateStateProbabilities(int T, Matrix* state_probabilities) {
    
    
    state_probabilities -> Init(n_states_, T);
    
    // base case
    for(int j = 0; j < n_states_; j++) {
      state_probabilities -> set(j, 0, p_initial_[j]);
    }
    
    // recursive step
    for(int t = 1; t < T; t++) {
      for(int j = 0; j < n_states_; j++) {
	double sum = 0;
	for(int i = 0; i < n_states_; i++) {
	  sum += state_probabilities -> get(i, t - 1) * p_transition_.get(i, j);
	}
	state_probabilities -> set(j, t, sum);
      }
    }
  }

};


#endif /* HMM */
