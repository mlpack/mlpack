
#include <fastlib/fastlib.h>
#include "pa.h"

// Marco to append the result file, should be used only once
// because it appends a time identification for each result output
#define PRINT_RESULT( x ) \
  { FILE* f = fopen("result.txt", "a+"); \
    time_t seconds = time(NULL); \
    fprintf(f, "%08x: ", (unsigned int) seconds); \
    x \
    fclose(f); }

const fx_entry_doc patest_entries[] = {
  
  {"data", FX_REQUIRED, FX_STR, NULL,
   " data file consists of data points and theirs labels.\n"},
  //{"relErr", FX_PARAM, FX_DOUBLE, NULL,
  // " Target relative error |A|-|A'|/|A|, default = 0.1.\n"},
  //{"U_out", FX_PARAM, FX_STR, NULL,
  // " File to hold matrix U.\n"},
  //{"s_out", FX_PARAM, FX_STR, NULL,
  // " File to hold the singular values vector s.\n"},
  //{"VT_out", FX_PARAM, FX_STR, NULL,
  // " File to hold matrix VT (V transposed).\n"},
  //{"SVT_out", FX_PARAM, FX_STR, NULL,
  // " File to hold matrix S * VT (the dimension reduced data).\n"},
  //{"lasvd", FX_PARAM, FX_STR, NULL,
  // " Use this parameter to compare running time to that of la::SVDInit().\n"},
  //{"quicsvd_time", FX_TIMER, FX_CUSTOM, NULL,
  // " time to run the QUIC-SVD algorithm.\n"},
  //{"lasvd_time", FX_TIMER, FX_CUSTOM, NULL,
  // " time to run the SVD algorithm from LAPACK.\n"},
  {"avg_error", FX_RESULT, FX_DOUBLE, NULL,
   " average error over sequence.\n"},
  {"avg_loss", FX_RESULT, FX_DOUBLE, NULL,
   " average loss over sequence.\n"},
  //{"dimension", FX_RESULT, FX_INT, NULL,
  // " the reduced dimension of the data.\n"},
  
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc patest_submodules[] = {
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc patest_doc = {
  patest_entries, patest_submodules,
  "This is a program testing Passive Aggressive algorithm "  
  "and its variants.\n"
};

void Run_PA(Dataset& data, Vector& w_out, 
	    double& avg_error_out, double& avg_loss_out) {
  //printf("data.matrix.n_rows = %d\n", data.matrix().n_rows());
  //printf("n_points = %d\n", data.n_points());
  //printf("n_features = %d\n", data.n_features());
  //printf("n_labels = %d\n", data.n_labels());
  w_out.Init(data.n_features()-1);
  w_out.SetZero();

  avg_error_out = 0;
  avg_loss_out = 0;
  for (int i_p = 0; i_p < data.n_points(); i_p++) {
    double* X_t = data.point(i_p);
    double y_t = data.get(data.n_features()-1, i_p);
    Vector w_tmp;
    double loss_t = PA_Update(w_out, X_t, y_t, w_tmp);

    if (loss_t > 1) avg_error_out += 1.0;
    avg_loss_out += loss_t;

    w_out.CopyValues(w_tmp);
  }
  avg_loss_out /= data.n_points();
  avg_error_out /= data.n_points();
}

void Run_PA_I(Dataset& data, Vector& w_out, 
	      fx_module* module) {
  //printf("data.matrix.n_rows = %d\n", data.matrix().n_rows());
  //printf("n_points = %d\n", data.n_points());
  //printf("n_features = %d\n", data.n_features());
  //printf("n_labels = %d\n", data.n_labels());
  w_out.Init(data.n_features()-1);
  w_out.SetZero();

  double C = fx_param_double(module, "C", 0.001);
  double avg_error = 0;
  double avg_loss = 0;
  for (int i_p = 0; i_p < data.n_points(); i_p++) {
    double* X_t = data.point(i_p);
    double y_t = data.get(data.n_features()-1, i_p);
    Vector w_tmp;
    double loss_t = PA_I_Update(w_out, X_t, y_t, w_tmp);

    if (loss_t > 1) avg_error_out += 1.0;
    avg_loss_out += loss_t;

    w_out.CopyValues(w_tmp);
  }
  avg_loss /= data.n_points();
  avg_error /= data.n_points();

  fx_result_double(module, "avg_error", avg_error);
  fx_result_double(module, "avg_loss", avg_loss);
}


int main(int argc, char** argv) {
  fx_module *root = fx_init(argc, argv, &patest_doc);  
  
  const char* filename = fx_param_str_req(root, "data");
  Dataset data;

  if (data.InitFromFile(filename) != SUCCESS_PASS) {
    PRINT_RESULT( fprintf(f, "data = %s error read file\n", filename); );
    exit(1);
  }

  Vector weight;
  double avg_error, avg_loss;

  /* 
  Run_PA(data, weight, avg_error, avg_loss);
  PRINT_RESULT(
    fprintf(f, "data = %s PA avg_error = %e avg_loss = %e\n", 
	    filename, avg_error, avg_loss);	  
  )
  */

  Run_PA_I(data, weight, root);
  PRINT_RESULT(
    fprintf(f, "data = %s PA_I C = %f avg_error = %e avg_loss = %e\n", 
	    filename, 
	    fx_param_double(root, "C", 0.001);
	    avg_error, 
	    avg_loss);	  
  )

  fx_result_double(root, "avg_error", avg_error);
  fx_result_double(root, "avg_loss", avg_loss);


  fx_done(root);
}
