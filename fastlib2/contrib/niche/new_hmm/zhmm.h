#ifndef HMM_H
#define HMM_H


#include "fastlib/fastlib.h"
#include "distribution.h"
#include "gaussian.h"

#define MULTINOMIAL 0
#define GAUSSIAN 1
#define MIXTURE 2


template <typename TDistribution>
class HMM {

 private:

  int n_states_;
  int n_dims_;

 public:

  // p_initial, p_transition, and state_distributions are public because
  // it's ever so much easier

  Vector p_initial;

  /* rows sum to 1: p_transition_.get(i,j) is P(s_j | s_i) */
  Matrix p_transition; //

  TDistribution* state_distributions;
  
    

 public:

  void Init(int n_states_in, int n_dims_in) {
    n_states_ = n_states_in;
    n_dims_ = n_dims_in;
    
    p_initial.Init(n_states_);
    p_transition.Init(n_states_, n_states_);
    
    state_distributions = 
      (TDistribution*) malloc(n_states_ * sizeof(TDistribution));
    for(int i = 0; i < n_states_; i++) {
      state_distributions[i].Init(n_dims_);
    }
  }

  /*  
      void RandomlyInitialize() {
      double uniform = ((double) 1) / ((double) n_states_);

      for(int i = 0; i < n_states_; i++) {
      p_initial[i] = uniform;
      for(int j = 0; j < n_states_; j++) {
      p_transition.set(j, i, uniform);
      }
   
      state_distributions[i].RandomlyInitialize();
      }
      }
  */

  void BaumWelch(ArrayList <Matrix> sequences>) {

    // First, we declare an HMM that we can use during EM
    HMM<TDistribution> new_hmm.Init(n_states_, n_dims_);
    
    // used for computing diagonal covariance below
    Vector result;
    result.Init(n_dims_);
    
    // recycling is good. let's use this repeatedly
    Vector new_p_transition_denom;
    new_p_transition_denom.Init(n_states_);

    Vector gaussian_denom;
    if(GAUSSIAN) {
      gaussian_denom.Init(n_states_);
    }
    else {
      gaussian_denom.Init(0);
    }

    Vector weight_qi;
    
    if(MIXTURE) {
      weight_qi.Init(n_states_);
    }
    else {
      weight_qi.Init(0);
    }



    bool converged = false;

    while(!converged) {

      new_hmm.p_initial.SetZero();
      new_hmm.p_transition.SetZero();
      new_hmm_p_transition_denom.SetZero();
      if(DISCRETE) {
	for(int i = 0; i < n_states_; i++) {
	  new_hmm.state_distributions[i] -> p.SetZero();
	}
      }
      else if(GAUSSIAN) {
	for(int i = 0; i < n_states_; i++) {
	  new_hmm.state_distributions[i] -> mu.SetZero();
	  new_hmm.state_distributions[i] -> sigma.SetZero();
	}
	gaussian_denom.SetZero();
      }
      else if(MIXTURE) {
	for(int i = 0; i < n_states_; i++) {
	  new_hmm.state_distributions[i].weights.SetZero();
	  for(int k = 0; k < n_mixtures_; k++) {
	    new_hmm.state_distributions[i].components[k] -> mu.SetZero();
	    new_hmm.state_distributions[i].components[k] -> sigma.SetZero();
	  }
	}
	weight_qi.SetZero();
      }

      for(int m = 0; m < n_sequences; m++) {
	Matrix &sequence = sequences[m];
	int sequence_length = sequence.n_cols();

	Matrix p_x_given_q; // Rabiner's b
	ArrayList<Matrix> p_x_given_mixture_q; // P_qik(x_t) / P_qi(x_t)
	ArrayList<Matrix> p_qq_t; // Rabiner's xi
	Matrix p_qt; // Rabiner's gamma
	double neg_likelihood = 1;

	ExpectationStep(sequence,
			&p_x_given_q, &p_x_given_mixture_q,
			&p_qq_t, &p_qt,
			&neg_likelihood);
      

	/////////////////////////////
	// M STEP
	/////////////////////////////
     
	// ACCUMULATE
       
	// accumulate initial state probabilities
	for(int i = 0; i < n_states_; i++) {
	  new_hmm.p_initial[i] += p_qt(0, i);
	}
      
	// accumulate state transition probabilities
	for(int i = 0; i < n_states_; i++) {
	  for(int t = 0; t < sequence_length - 1; t++) {
	    for(int j = 0; j < n_states_; j++) {
	      // Note that new_hmm.p_transition is treated as its transpose
	      // initially for efficiency. We transpose it at the end of
	      // its computation.
	      new_hmm.p_transition(j, i,
				   new_hmm.p_transition.get(j, i)
				   + p_qq_t[i].get(j, t));
	    }
	    new_hmm_p_transition_denom[i] += p_qt.get(t, i); 
	  }
	}

	// accumulate density statistics for observables
	if(DISCRETE) {
	  for(int i = 0; i < n_states_; i++) {
	    Vector& new_p = new_hmm.state_distributions[i] -> p;
	    for(int t = 0; t < sequence_length; t++) {
	      new_p[sequence.get(0, t)] += p_qt.get(t, i);
	    }
	  }
	}
	else if(GAUSSIAN) {
	  for(int i = 0; i < n_states_; i++) {
	    Vector &new_mu = new_hmm.state_distributions[i] -> mu;
	    Vector &new_sigma = new_hmm.state_distributions[i] -> sigma;
	    for(int t = 0; t < sequence_length; t++) {
	      Vector x_t;
	      sequence.MakeColumnVector(t, &x_t);

	      // update mean
	      la::AddExpert(p_qt.get(t, i), x_t, &new_mu);

	      // update covariance
	      la::SubOverwrite(state_distributions[i] -> mu, x_t, &result);
	      for(int j = 0; j < n_dims_; j++) {
		result[j] *= result[j];
	      }
	      la::AddExpert(p_qt.get(t, i), result, &new_sigma);

	      // keep track of normalization factor
	      gaussian_denom[i] += p_qt.get(t, i);
	    }
	  }
	}
	else if(MIXTURE) {
	  for(int k = 0; k < n_components_; k++) {
	    for(int i = 0; i < n_states_; i++) {
	      Vector &new_mu =
		new_hmm.state_distributions[i].components[k] -> mu;
	      Vector &new_sigma =
		new_hmm.state_distributions[i].components[k] -> sigma;
	      double sum = new_state_distributions[i].weights[k];
	      for(int t = 0; t < sequence_length; t++) {
		Vector x_t;
		sequence.MakeColumnVector(t, &x_t);

		// hopefully p_xt_given_mixture_q[k].get(i,t) is fast since
		// n_states_ is usually small (< 20)
		double scaling_factor =
		  p_qt.get(t, i) * p_xt_given_mixture_q[k].get(i, t);
	      
		sum += scaling_factor;

		// update mean
		la::AddExpert(scaling_factor, x_t, &new_mu);

		// update covariance
		la::SubOverwrite(state_distributions[i].components[k] -> mu,
				 x_t, &result);
		for(int j = 0; j < n_dims_; j++) {
		  result[j] *= result[j];
		}
		la::AddExpert(scaling_factor, result, &new_sigma);
	      
		// keep track of normalization factor
		//gaussian_denom[i] += p_qt.get(t, i);
	      }
	      new_hmm.state_distributions[i].weights[k] = sum;
	    }

	    // note that new_hmm_p_transition_denom[i] =
	    //             \sum_{t = 0 -> T - 2} p_qt.get(t, i)
	    weight_qi[i] +=
	      new_hmm_p_transition_denom[i] + p_qt.get(n_sequences - 1, i);
	    // so, weight_qi[i] = \sum_{t = 0 -> T - 1} p_qt.get(t, i)
	  
	  }
	} //end if(MIXTURE)
      } // end for(all sequences)


      // ACCUMULATION COMPLETE, NORMALIZE THE WORLD!

      // normalize initial state probabilities
      la::Scale(1/ Sum(new_hmm.p_initial), &(new_hmm.p_initial));

      // normalize state transition probabilities
      for(int i = 0; i < n_states_; i++) {
	for(int j = 0; j < n_states_; j++) {
	  // Again, Note that new_hmm.p_transition is treated as its transpose
	  // initially for efficiency. We transpose it at the end of
	  // its computation (below).
	  new_hmm.p_transition.set(j, i,
				   new_hmm.p_transition.get(j, i)
				   / new_hmm_p_transition_denom[i]);
	}
      }
      la::TransposeSquare(&(new_hmm.p_transition));

      // normalize density statistics for observables
      if(DISCRETE) {
	for(int i = 0; i < n_states_; i++) {
	  Vector& new_p = new_hmm.state_distributions[i] -> p;
	  la::Scale(((double)1) / Sum(new_p),
		    &new_p);
	}
      }
      else if(GAUSSIAN) {
	for(int i = 0; i < n_states_; i++) {
	  double normalization_factor = ((double)1) / gaussian_denom[i];
	  Vector &new_mu = new_hmm.state_distributions[i] -> mu;
	  la::Scale(normalization_factor,
		    &new_mu);
	  Vector &new_sigma = new_hmm.state_distributions[i] -> sigma;
	  la::Scale(normalization_factor,
		    &new_sigma);
	}
      }
      else if(MIXTURE) {
	for(int k = 0; k < n_components_; k++) {
	  for(int i = 0; i < n_states_; i++) {
	    double normalization_factor =
	      ((double)1) / new_hmm.state_distributions[i].weights[k];
	    Vector &new_mu =
	      new_hmm.state_distributions[i].components[k] -> mu;
	    la::Scale(normalization_factor,
		      &new_mu);
	    Vector &new_sigma =
	      new_hmm.state_distributions[i].components[k] -> sigma;
	    la::Scale(normalization_factor,
		      &new_sigma);
	  
	    new_hmm.state_distributions[i].weights[k] /= weight_qi[i];
	  }
	}
      }

      // SWAP new and old HMMs' initial state probabilities
      //                        state transition probabilities
      //                        state distributions
      TDistribution* temp_state_distributions;
      temp_state_distributions = state_distributions;
      state_distributions = new_hmm.state_distributions;
      new_hmm.state_distributions = temp_state_distributions;

      Vector temp_p_initial;
      temp_p_initial.Own(&p_initial);
      p_initial.Own(&(new_hmm.p_initial));
      new_hmm.p_initial.Own(&temp_p_initial);

      Matrix temp_p_transition;
      temp_p_transition.Own(&p_transition);
      p_transition.Own(&(new_hmm.p_transition));
      new_hmm.p_transition.Own(&temp_p_transition);
  
      // How far have we come? Have we converged? Is there a Buddha?
    
    } // end while(!converged)
  }



  void ExpectationStep(const Matrix &sequence,
		       Matrix* p_p_x_given_q,
		       ArrayList<Matrix>* p_p_x_given_mixture_q,
		       ArrayList<Matrix>* p_p_qq_t
		       Matrix* p_p_qt,
		       double* p_neg_likelihood) {
    
    // embrace readability!
    Matrix &p_x_given_q = *p_p_x_given_q;
    ArrayList<Matrix> &p_x_given_mixture_q = *p_p_x_given_mixture_q;
    ArrayList<Matrix> &p_qq_t = *p_p_qq_t;
    Matrix &p_qt = *p_p_qt;
    double &neg_likelihood = *p_neg_likelihood;

    if(MIXTURE) {
      ComputePxGivenMixtureQ(sequence, &p_x_given_q,
			     &p_x_given_mixture_q);
    }
    else {
      ComputePxGivenQ(sequence, &p_x_given_q);
      p_x_given_mixture_q.Init(0);
    }
    
    Matrix forward_vars;
    Matrix backward_vars;
    Vector scaling_vars;
    ForwardAlgorithm(p_x_given_q, &scaling_vars, &forward_vars);
    BackwardAlgorithm(p_x_given_q, scaling_vars, &backward_vars);
    
    ComputePqqt(forward_vars, backward_vars, p_x_given_q, &p_qq_t);
    ComputePqt(forward_vars ,backward_vars, scaling_vars, p_qq_t, &p_qt);

    neg_likelihood = 0;
    int sequence_length = sequence.n_cols();
    for(int i = 0; i < sequence_length; i++) {
      neg_likelihood -= log(scaling_vars[i]);
    }
  }


  void ComputePxGivenQ(const Matrix &sequence,
		       Matrix* p_p_x_given_q) {
    Matrix &p_x_given_q = *p_p_x_given_q;
    
    int sequence_length = sequence.n_cols();
    p_x_given_q.Init(n_states_, sequence_length);
    
    for(int t = 0; t < sequence_length; t++) {
      Vector x_t;
      sequence.MakeColumnVector(t, &x_t);
      
      for(int i = 0; i < n_states_; i++) {
	p_x_given_q.set(i, t, state_distributions_[i].pdf(x_t));
	// ensure that class member function exists for classes:
	//                                                       Multinomial
	//                                                       Gaussian
	//                                                       Mixture
      }
    }
  }

  void ComputePxGivenMixtureQ(const Matrix &sequence,
			      Matrix* p_p_x_given_q,
			      ArrayList<Matrix>* p_p_x_given_mixture_q) {
    Matrix &p_x_given_q = *p_p_x_given_q;
    ArrayList<Matrix> &p_x_given_mixture_q = *p_p_x_given_mixture_q;
    
    int sequence_length = sequence.n_cols();
    p_x_given_q.Init(n_states_, sequence_length);
    p_x_given_q.SetZero();

    p_x_given_mixture_q.Init(n_components_);
    for(int k = 0; k < n_components_; k++) {
      p_x_given_mixture_q[k].Init(n_states_, sequence_length);
    }

    for(int k = 0; k < n_components_; k++) {
      for(int t = 0; t < sequence_length; t++) {
	Vector xt;
	sequence.MakeColumnVector(t, &xt);

	for(int i = 0; i < n_states_; i++) {
	  double p_xt_given_qik = state_distributions_[i].PkthComponent(k, xt);
	  p_x_given_mixture_q[k].set(i, t,
				     p_xt_given_qik);
	  p_x_given_q.set(i, t,
			  p_x_given_q.get(i, t) + p_xt_given_qik);
	}
      }
    }

    for(int k = 0; k < n_components_; k++) {
      for(int t = 0; t < sequence_length; t++) {
	for(int i = 0; i < n_states_; i++) {
	  p_x_given_mixture_q[k].set(i, t,
				     p_x_given_mixture_q[k].get(i, t)
				     / p_x_given_q.get(i, t));
	}
      }
    }
  }


  void ForwardAlgorithm(const Matrix &p_x_given_q,
			Vector* p_scaling_vars, 
			Matrix* p_forward_vars) {
    Vector& scaling_vars = *p_scaling_vars;
    Matrix& forward_vars = *p_forward_vars;

    int sequence_length = p_x_given_q.n_cols();

    scaling_vars.Init(sequence_length);
    forward_vars.Init(n_states_, sequence_length);

    Vector forward_0;
    forward_vars.MakeColumnVector(0, &forward_0);

    Vector p_x0_given_q;
    p_x_given_q.MakeColumnVector(0, &p_x0_given_q);

    HadamardMultiplyOverwrite(p_initial_, p_x0_given_q, &forward_0);
    ScaleForwardVar(&(c[0]), &forward_0);

    for(int t = 0; t < T - 1; t++) {
      Vector forward_t;
      forward_vars.MakeColumnVector(t, &forward_t);

      Vector forward_t_plus_1;
      forward_vars.MakeColumnVector(t, &forward_t_plus_1);

      Vector p_xt_plus_1_given_q;
      p_x_given_q.MakeColumnVector(t + 1, p_xt_plus_1_given_q);

      la::MulOverwrite(forward_t, p_transition_, &forward_t_plus_1);
      HadamardMultiplyBy(p_xt_plus_1_given_q, &forward_t_plus_1);
      ScaleForwardVar(&(c[t + 1]), &forward_t_plus_1);
    }
  }

  void BackwardAlgorithm(const Matrix &p_x_given_q,
			 const Vector &scaling_vars,
			 Matrix *p_backward_vars) {
    Matrix& backward_vars = *p_backward_vars;
    
    int sequence_length = p_x_given_q.n_cols();
    
    backward_vars.Init(n_states_, sequence_length);
    
    Vector backward_T_minus_1;
    backward_vars.MakeColumnVector(sequence_length - 1, &backward_T_minus_1);
    
    backward_T_minus_1.SetAll(c[sequence_length - 1]);

    for(int t = sequence_length - 2; t >= 0; t--) {
      Vector backward_t;
      backward_vars.MakeColumnVector(t, &backward_t);

      Vector backward_t_plus_1;
      backward_vars.MakeColumnVector(t + 1, &backward_t_plus_1);

      Vector p_xt_plus_1_given_q;
      p_x_given_q.MakeColumnVector(t + 1, &p_xt_plus_1_given_q);

      Vector result;
      HadamardMultiplyInit(p_xt_plus_1_given_q, backward_t_plus_1,
			   &result);
      la::MulInit(p_transition_, result, &backward_t);
      la::Scale(c[t], &backward_t);
    }
  }


  void ComputePqqt(const Matrix &forward_vars,
		   const Matrix &backward_vars,
		   const Matrix &p_x_given_q,
		   ArrayList<Matrix>* p_p_qq_t);
  ArrayList<Matrix> &p_qq_t = *p_p_qq_t;
  
  int sequence_length = forward_vars.n_cols();
  
  p_qq_t.Init(n_states_);
  for(int i = 0; i < n_states_; i++) {
    p_qq_t[i].Init(n_states_, sequence_length);
    
    for(int t = 0; t < sequence_length - 1; t++) {
      for(int j = 0; j < n_states_; j++) {
	//consider transposing forward_vars and p_transition_ for efficiency
	p_qq_t[i].set(j, t,
		      forward_vars.get(i, t)
		      * p_transition_.get(i, j)
		      * p_x_given_q.get(j, t + 1)
		      * backward_vars.get(j, t + 1));
      }
    }
  }

  void ComputePqt(const Matrix &forward_vars,
		  const Matrix &backward_vars,
		  const Vector &scaling_vars,
		  const ArrayList<Matrix> &p_qq_t,
		  Matrix* p_p_qt) {
    Matrix &p_qt = *p_p_qt;

    int sequence_length = forward_vars.length();

    p_qt.Init(sequence_length, n_states_);

    for(int i = 0; i < n_states_; i++) {
      for(int t = 0; t < sequence_length; t++) {
	p_qt.set(t, i, 
		 forward_vars.get(i, t)
		 * backward_vars.get(i, t)
		 / scaling_vars.get(t));
      }
    }
  }

  void HadamardMultiplyInit(const Vector &x,
			    const Vector &y,
			    Vector* z) {
    z -> Init(x.length());
    for(int i = 0; i < x.length(); i++) {
      (*z)[i] = x[i] * y[i];
    }
  }

  void HadamardMultiplyOverwrite(const Vector &x,
				 const Vector &y,
				 Vector* z) {
    for(int i = 0; i < x.length(); i++) {
      (*z)[i] = x[i] * y[i];
    }
  }
 
  void HadamardMultiplyBy(const Vector &x,
			  Vector* y) {
    for(int i = 0; i < x.length(); i++) {
      (*y)[i] *= x[i];
    }
  }

  double Sum(const Vector &x) {
    double sum = 0;
    for(int i = 0; i < x.length(); i++) {
      sum += x[i];
    }
    return sum;
  }
 
  void ScaleForwardVar(double* c, Vector *p_forward_var) {
    (*c) = ((double)1) / Sum(*p_forward_var);
    la::Scale(*c, p_forward_var);
  }
 




  void SetStateDistribution(int i,
			    const TDistribution &distribution) {
    state_distributions_[i].CopyValues(distribution);
  }


  // accessors

  int n_states() const {
    return n_states_;
  }

  int n_dims() const {
    return n_dims_;
  }
  
  Vector p_initial() const {
    return p_initial_;
  }

  Matrix p_transition() const {
    return p_transition_;
  }

  TDistribution* state_distributions() const {
    return state_distributions_;
  }

  void PrintDebug(const char *name = "", FILE *stream = stderr) const {
    fprintf(stream, "----- HMM %s ------\n", name);
    
    p_initial_.PrintDebug("initial probabilities", stream);
    p_transition_.PrintDebug("transition probabilities", stream);

    char string[100];    
    for(int i = 0 ;i < n_states_; i++) {
      sprintf(string, "state %d:\n", i+1);
      state_distributions_[i].PrintDebug(string);
      fprintf(stream, "\n");
    }
  }

  void SetPInitial(Vector p_initial_in) {
    p_initial_.CopyValues(p_initial_in);
  }

  void SetPTransition(Matrix p_transition_in) {
    p_transition_.CopyValues(p_transition_in);
  }
  

  ~HMM() {
    //printf("destroying HMM\n");
    Destruct();
  }

  void Destruct() {
    for(int i = 0; i < n_states_; i++) {
      state_distributions_[i].Destruct();
    }
    free(state_distributions_);
  }

};


#endif /* HMM_H */
