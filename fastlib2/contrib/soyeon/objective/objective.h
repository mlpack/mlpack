#include "fastlib/fastlib.h"


class Objective {
 public:
  void Init(fx_module *module);
  void ComputeObjective(Matrix &x, double *value);
 
 private:
  ArrayList<Matrix> first_stage_x_; 
  // If the value is -1 then it corresponds
  // to all zeros in y
  // If it is greter than zero then it corresponds to 
  // the non zero element index
  ArrayList<index_t>  first_stage_y_;
  double ComputeTerm1_();
  double ComputeTerm2_();
  double ComputeTerm3_();
};

