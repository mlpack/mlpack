#ifndef OBJ_GRAD_HESS_CLASS_H_
#define OBJ_GRAD_HESS_CLASS_H_
#include "fastlib/fastlib.h"

class ObjectGradientHessian {
 public:
  void Init(int &num_points_t); 
  void Destruct();
	void ApproxBetafnNominator(int &num_points_t, double *beta_nominator);
 
 
private:
	int num_points_t_;
  //memeber variable with _(under score)

	//Matrix *Xmatrix_;
	//Matrix *unk_Xmatrix_;
	
	//int m;	//m points for t

	/*int	num_attribute_;
	int	num_unk_attribute_;
	int	npar_;
	int	num_ppl_;
	int	num_selected_ppl_;
	int	*ppl_selector_;
	int	*num_alternatives_;
	
	double *param_;
	double *param_new_;

  //Vector	*random_alpha_;
	//Vector	*random_t_;
	Matrix	dot_Xn1_;
	Matrix	dot_Xn2_;
	Matrix	unk_attribute_past_;
	*/

};



#endif

