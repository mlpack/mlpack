#include "test_obj.h"

void RosenbrockFunction::Init(fx_module *module) {
  module_=module;
}

void RosenbrockFunction::ComputeObjective(Vector &x, double *objective) {
  DEBUG_SAME_SIZE(x.length(), quadratic_term_.n_rows()); 
  Vector temp1;
  la::MulInit(x, quadratic_term_, &temp1);
  *objective = la::Dot(temp1, x) + la::Dot(linear_term_, x);
}

void RosenbrockFunction::ComputeGradient(Vector &x, Vector *gradient){
  
  //Matrix sum_quad;
  Vector temp2;
  la::MulInit(sum_quad_, x, &temp2);
  la::AddInit(linear_term_, temp2, gradient);
}

void RosenbrockFunction::ComputeHessian(Vector &x, Matrix *hessian){
	//Be careful!!
  hessian = &sum_quad_; //hessian: address, *hessian: value at address
	//*hessian = sum_quad_;
	
}





