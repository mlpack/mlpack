#ifndef NAIVE_MULTIBODY_H
#define NAIVE_MULTIBODY_H

template<typename TMultibodyKernel>
class NaiveMultibody {

  FORBID_ACCIDENTAL_COPIES(NaiveMultibody);

 private:

  /** Temporary space for storing indices selected for exhaustive computation
   */
  ArrayList<int> exhaustive_indices_;

  /** dataset for the tree */
  Matrix data_;

  /** multibody kernel function */
  TMultibodyKernel mkernel_;

  /** potential estimate */
  double neg_potential_e_;
  double pos_potential_e_;

  /** exhaustive computer */
  void NMultibody(int level) {
    
    int num_nodes = mkernel_.order();
    int start_index = 0;
    double neg, pos;

    if(level < num_nodes) {
      
      if(level == 0) {
	start_index = 0;
      }
      else {
	start_index = exhaustive_indices_[level - 1] + 1;
      }
      
      for(index_t i = start_index; i < data_.n_cols() - 
	    (num_nodes - level - 1); i++) {
	exhaustive_indices_[level] = i;
	NMultibody(level + 1);
      }
    }
    else {
      mkernel_.Eval(data_, exhaustive_indices_, &neg, &pos);
      neg_potential_e_ += neg;
      pos_potential_e_ += pos;
    }
  }

 public:

  NaiveMultibody() {}
  
  ~NaiveMultibody() {}

  void Init(const Matrix &data, double bandwidth) {
    data_.Alias(data);
    exhaustive_indices_.Init(3);
    mkernel_.Init(bandwidth);
    neg_potential_e_ = pos_potential_e_ = 0;
  }

  void Compute() {

    NMultibody(0);

    printf("Negative potential sum %g\n", neg_potential_e_);
    printf("Positive potential sum %g\n", pos_potential_e_);
    printf("Got potential sum %g\n", neg_potential_e_ + pos_potential_e_);
  }

};

#endif
