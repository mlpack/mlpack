/*
 * =====================================================================================
 *
 *       Filename:  optim_test.cc
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  11/11/2008 10:52:49 PM EST
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 *
 * =====================================================================================
 */

#include "optimizer.h"
class Rosen {
 public:
  Rosen() {
    dimension_ = 2;
  };
  ~Rosen(){}; 
  void GiveInit(Vector *vec) {
    (*vec)[0]=0;
    (*vec)[1]=1; 
  }
  void ComputeObjective(Vector &x, double *value) {
     double x1=x[0];
     double x2=x[1];
     double f1=(x2-x1*x1);
     double f2=1.-x1;
     *value = 100. *f1*f1+f2*f2;
   } 
   void ComputeGradient(Vector &x, Vector *gx) {
     double x1=x[0];
     double x2=x[1];
     double f1=(x2-x1*x1);
     double f2=1.-x1;
     (*gx)[0]=-400.*f1*x1-2.*f2;
     (*gx)[1]=200.*f1;
   }
   index_t dimension() {
     return dimension_;
   } 
 private:
  index_t dimension_;
    
};


class StaticOptppOptimizerTest {
 public:
  StaticOptppOptimizerTest(fx_module *module) {
    module_ = module;
  }
  void Test1() {
    Rosen rosen;
    optimizer_.Init(module_, &rosen);
    Vector result;
    optimizer_.Optimize(&result);  
  }
  void TestAll() {
    Test1();
  }
 private:
  fx_module *module_;
  optim::optpp::StaticOptppOptimizer<optim::optpp::LBFGS, Rosen> optimizer_;
};

int main(int argc, char *argv[]) {
  fx_module *fx_root = fx_init(argc, argv, NULL);
  StaticOptppOptimizerTest test(fx_root);
  test.TestAll();
  fx_done(fx_root);
  return 0;
}
