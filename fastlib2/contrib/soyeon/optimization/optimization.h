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
		/*
		void ComputeSteihaugDirection(double radius, 
																Vector &gradient,
																Matrix &hessian,
																Vector *p,
																double *delta_m);
																*/




	private:
		fx_module *module_;
		




};


