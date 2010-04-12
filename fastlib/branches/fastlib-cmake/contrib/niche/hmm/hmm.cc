#include<fastlib/fastlib.h>
#include<stdlib.h>

/* state prior probabilities */

/* state transition probabilities */

/* observation densities */
// autoregressive


// 1st - read in a set of N training sequences X_i of length T_i
// this should be an array list of vectors X_i of length T_i

// impose conditional independencies on state transition matrix
// initialize parameters - state segmentation

// the observations should be loosely labeled into states

/* 
   1) Choose graphical model structure
     A) hierarchical state model - Rest submodel and Active submodel
     B) number of states in each submodel
     C) permissible state transitions
       * each model is left-right
       * each model has one start state and one end state
   2) Given training data, segment it
     A) segment by submodel using ground-truth (this data is Active class data)
     B) using some method of clustering to segment a submodel's data into
        states - segmental K-means clustering using AR coefficients
   3) Calculate initial priors, transition probs, and AR paramaters
   4) Calculate forward and backward probs in order to get posterior probs
   5) Use posterior probs to update prior and transition probs
   6) Update autocorrelation using posterior probs and previous autocorrelation
   7) Solve linear system to get AR parameters
   8) Calculate autocorrelation of autoregressive coefficients
*/


class HMM {
  Vector prior_probs; // pi_i
  Matrix trans_probs; // a_{i,j}
  Vector r; // autocorrelation (AR model sufficient statistics)
  Vector posterior_probs; // gamma
  Matrix joint_posterior_probs; // xi
};




int main(int argc, char* argv[]) {

  fx_init(argc, argv);

  



  fx_done();

  return SUCCESS_PASS;

}


  
