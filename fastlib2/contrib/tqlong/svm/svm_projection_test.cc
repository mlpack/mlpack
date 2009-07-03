#include <fastlib/fastlib.h>
#include <iostream>
#include <list>
#include "svm_projection.h"

using namespace SVM_Projection;

const fx_entry_doc svm_project_main_entries[] = {
  {"n", FX_PARAM, FX_INT, NULL,
   " Number of data points. \n"},
  {"ns", FX_PARAM, FX_INT, NULL,
   " Size of subset of variables. \n"},
  {"print", FX_PARAM, FX_STR, NULL,
   " Print the kernel matrix. \n"},
  {"verbose", FX_PARAM, FX_STR, NULL,
   " Print iteration information.\n"},
  {"svm_projection_time", FX_TIMER, FX_CUSTOM, NULL,
   " Run time.\n"},
  {"SVs", FX_RESULT, FX_INT, NULL,
   " Number of support vectors.\n"},
  {"input", FX_PARAM, FX_STR, NULL,
   " File consists of the kernel matrix.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc svm_project_main_submodules[] = {
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc svm_project_main_doc = {
  svm_project_main_entries, svm_project_main_submodules,
  "This is a program calculating SVM using projection technique. \n"
};


int main(int argc, char** argv) {
  fx_init(argc, argv, &svm_project_main_doc);
  SV_index SVs;

  /*
  Vector c, x;
  c.Init(3);
  c[0] = 1.0; c[1] = 4.0; c[2] = 1.0;
  x.Copy(c);

  ProjectOnSimplex(c, &x, &SVs);

  ot::Print(x);
  */

  Matrix Q;
  
  printf("Prepare data ...\n"); fflush(stdout);
  if (fx_param_exists(NULL, "input")) {
    const char* fn = fx_param_str(NULL, "input", NULL);
    data::Load(fn, &Q);
  }
  else {
    Matrix A;
    index_t n = fx_param_int(NULL, "n", 10);
    A.Init(n, n);
    for (index_t i = 0; i < n; i++)
      for (index_t j = 0; j < n; j++) A.ref(i, j) = math::Random();
    la::MulTransAInit(A, A, &Q);
    //Q.Init(n,n); Q.SetZero();
    //for (int i = 0; i < n; i++) Q.ref(i,i) = 1.0;
    //Q.ref(0, 0) = 2.0; Q.ref(1, 1) = 5.0;
  }

  if (fx_param_exists(NULL, "print")) ot::Print(Q);
  
  printf("n = %d\n", Q.n_rows()); fflush(stdout);

  Vector y;

  printf("Start SVM projection ...\n"); fflush(stdout);
  fx_timer_start(NULL, "svm_projection_time");
  //OptimizeQuadraticOnSimplex(Q, &y, &SVs);
  OptimQuadratic(Q, &y, &SVs);
  fx_timer_stop(NULL, "svm_projection_time");
  printf("done.\n");

  fx_result_int(NULL, "SVs", SVs.size());

  //ot::Print(y);

  fx_done(NULL);

  return 0;
}

