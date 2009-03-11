#include "fastlib/fastlib.h"

class OptimizationTest;
class Optimization {
	friend class OptimizationTest;
	public:
		void Init(fx_module *module);
		void ComputeDoglegDirection(double radius, 
																Vector &gradient,
																Matrix &hessian,
																Vector *p,
																double *delta_m);
		
		void ComputeSteihaugDirection(double radius, 
																	Vector &gradient,
																	Matrix &hessian,
																	Vector *p,
																	double *delta_m);

														




	private:
		fx_module *module_;
/*#define DBL_EPS 1e-16
#define ZERO_EPS 1e-9
#define PRINT_FREQ 1

#define MAXLINESIZE 1024
#define MAXFIELDSIZE 32
#define MAXFILENAMESIZE 64


* Constants for algorithm parameters.
 
#define ITERMAX 20000
#define RADIUS_MAX 10
#define RADIUS_INITIAL 0.001
#define NU 0.20
		*/


		




};


