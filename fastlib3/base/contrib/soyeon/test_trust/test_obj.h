#include "fastlib/fastlib.h"

class RosenbrockFunction {
 public:

	void Init(fx_module *module);
  //void Init(Matrix &quadratiic_term, Vector &linear_term); //Initialize the member data of a class
  void Destruct(); //decinstructor
  void ComputeObjective(Vector &x, double *objective);
  void ComputeGradient(Vector &x, Vector *gradient);
  void ComputeHessian(Vector &x, Matrix *hessian);
 
private:
  //memeber variable with _(under score)
	fx_module *module_;
  
};







