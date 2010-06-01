
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
   "Data file consists of data points and theirs labels.\n"},
  {"method", FX_REQUIRED, FX_STR, NULL,
   "Update scheme (PA, PA_I, PA_II).\n"},
  {"CV", FX_PARAM, FX_INT, NULL,
   "If equal to 1, do cross validation, default is 0.\n"},
  {"N", FX_PARAM, FX_INT, NULL,
   "Number of samples for cross validation.\n"},
  {"laps", FX_PARAM, FX_INT, NULL,
   "Number of laps, default is 1.\n"},
  {"C", FX_PARAM, FX_DOUBLE, NULL,
   "Penalty term for error, default 0.001.\n"},
  {"kernel", FX_PARAM, FX_STR, NULL,
   "Kernel type (linear, poly, gauss), default is linear. "
   "If not exist, use no-kernel version.\n"},
  {"order", FX_PARAM, FX_INT, NULL,
   "Polynomial kernel order, default is 2.\n"},
  {"homogeneous", FX_PARAM, FX_INT, NULL,
   "Is homogeneous polynomial kernel ? Default is 0.\n"},
  {"sigma", FX_PARAM, FX_DOUBLE, NULL,
  "Gaussian kernel width.\n"},
  {"avg_error", FX_RESULT, FX_DOUBLE, NULL,
   " average error over sequence.\n"},
  {"avg_loss", FX_RESULT, FX_DOUBLE, NULL,
   " average loss over sequence.\n"},
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

void Run_PA(fx_module* module, DataGenerator& data, Vector& w_out) {
  w_out.Init(data.n_features());
  w_out.SetZero();

  double (*update_func)(fx_module*, const Vector&, const Vector&, double, Vector&)
    = NULL;

  const char* method = fx_param_str_req(module, "method");
  if (strcmp(method, "PA") == 0) update_func = PA_Update;
  else if (strcmp(method, "PA_I") == 0) update_func = PA_I_Update;
  else if (strcmp(method, "PA_II") == 0) update_func = PA_II_Update;
  DEBUG_ASSERT(update_func != NULL);

  double avg_error = 0;
  double avg_loss = 0;
  
  for (;;) {
    Vector X_t;
    double y_t;
    if (!data.getNextPoint(X_t, y_t)) break;
    Vector w_tmp;
    double loss_t = update_func(module, w_out, X_t, y_t, w_tmp);

    if (loss_t > 1) avg_error += 1.0;
    avg_loss += loss_t;

    w_out.CopyValues(w_tmp);
    //printf("w[0] = %f\n", w_out[0]);
  }
  avg_loss /= data.n_points();
  avg_error /= data.n_points();

  printf("n_points = %d\n", data.n_points());
  printf("n_features = %d\n", data.n_features());

  fx_result_double(module, "avg_error", avg_error);
  fx_result_double(module, "avg_loss", avg_loss);
}

void Run_Kernelized_PA(fx_module* module, DataGenerator& data, KernelizedWeight& w) {
  double avg_error = 0;
  double avg_loss = 0;

  double (*update_func)(fx_module*, KernelizedWeight&, const Vector&, double) = NULL;

  const char* method = fx_param_str_req(module, "method");
  if (strcmp(method, "PA") == 0) update_func = Kernelized_PA_Update;
  else if (strcmp(method, "PA_I") == 0) update_func = Kernelized_PA_I_Update;
  else if (strcmp(method, "PA_II") == 0) update_func = Kernelized_PA_II_Update;
  DEBUG_ASSERT(update_func != NULL);
 
  for (;;) {
    Vector X_t;
    double y_t;
    if (!data.getNextPoint(X_t, y_t)) break;
    double loss_t = update_func(module, w, X_t, y_t);

    if (loss_t > 1) avg_error += 1.0;
    avg_loss += loss_t;
  }
  avg_loss /= data.n_points();
  avg_error /= data.n_points();

  fx_result_double(module, "avg_error", avg_error);
  fx_result_double(module, "avg_loss", avg_loss);
}

void CrossValidation(fx_module* module, DataGenerator& dg) {
  ArrayList<index_t> vIdx;
  // LOOCV
  index_t n_samples = fx_param_int(module, "N", 99);
  CrossValidationGenerator::createLOOCVindex(vIdx, n_samples);

  CrossValidationGenerator cvdg(dg, vIdx);

  double (*update_func)(fx_module*, const Vector&, const Vector&, double, 
			Vector&) = NULL;
  const char* method = fx_param_str_req(module, "method");
  if (strcmp(method, "PA") == 0) update_func = PA_Update;
  else if (strcmp(method, "PA_I") == 0) update_func = PA_I_Update;
  else if (strcmp(method, "PA_II") == 0) update_func = PA_II_Update;
  DEBUG_ASSERT(update_func != NULL);

  double cv_error = 0;
  for (index_t i_vSet = 0; i_vSet < cvdg.n_sets(); i_vSet++) {
    // training
    cvdg.setValidationSet(i_vSet, false);
    Vector w;
    w.Init(cvdg.n_features());
    w.SetZero();
    for (;;) {
      Vector X_t;
      double y_t;
      if (!cvdg.getNextPoint(X_t, y_t)) break;
      Vector w_tmp;
      update_func(module, w, X_t, y_t, w_tmp);
      w.CopyValues(w_tmp);
      //printf("P 1 %d\n", cvdg.n_points());
    }
    // Calculate training error
    double train_error = 0;
    cvdg.setValidationSet(i_vSet, false);
    for (;;) {
      Vector X_t;
      double y_t;
      if (!cvdg.getNextPoint(X_t, y_t)) break;
      double loss_t = hinge_loss(w, X_t, y_t);
      if (loss_t >= 1) train_error += 1.0;
    }
    train_error /= cvdg.n_points();
    // testing
    cvdg.setValidationSet(i_vSet, true);
    double avg_error = 0;
    for (;;) {
      Vector X_t;
      double y_t;
      if (!cvdg.getNextPoint(X_t, y_t)) break;
      double loss_t = hinge_loss(w, X_t, y_t);
      if (loss_t >= 1) avg_error += 1.0;
    }
    avg_error /= cvdg.n_points();
    printf("i_vSet = %d avg_error = %f train_error = %f cvdg.n_points = %d\n", 
	   i_vSet, avg_error, train_error, cvdg.n_points());
    cv_error += avg_error;
  }
  cv_error /= cvdg.n_sets();
  fx_result_double(module, "avg_error", cv_error); 
}
/** =============================================================
 */ 
void KernelizedCrossValidation(fx_module* module, KernelFunction& kernel, 
			       DataGenerator& dg) {
  ArrayList<index_t> vIdx;
  // LOOCV
  index_t n_samples = fx_param_int(module, "N", 99);
  CrossValidationGenerator::createLOOCVindex(vIdx, n_samples);

  CrossValidationGenerator cvdg(dg, vIdx);

  double (*update_func)(fx_module*, KernelizedWeight&, 
			const Vector&, double) = NULL;
  const char* method = fx_param_str_req(module, "method");
  if (strcmp(method, "PA") == 0) update_func = Kernelized_PA_Update;
  else if (strcmp(method, "PA_I") == 0) update_func = Kernelized_PA_I_Update;
  else if (strcmp(method, "PA_II") == 0) update_func = Kernelized_PA_II_Update;
  DEBUG_ASSERT(update_func != NULL);

  double cv_error = 0;
  for (index_t i_vSet = 0; i_vSet < cvdg.n_sets(); i_vSet++) {
    // training
    cvdg.setValidationSet(i_vSet, false);
    KernelizedWeight w(cvdg.n_features(), kernel);
    for (;;) {
      Vector X_t;
      double y_t;
      if (!cvdg.getNextPoint(X_t, y_t)) break;
      update_func(module, w, X_t, y_t);
    }
    // Calculate training error
    double train_error = 0;
    cvdg.setValidationSet(i_vSet, false);
    for (;;) {
      Vector X_t;
      double y_t;
      if (!cvdg.getNextPoint(X_t, y_t)) break;
      double loss_t = hinge_loss(w, X_t, y_t);
      if (loss_t >= 1) train_error += 1.0;
    }
    train_error /= cvdg.n_points();
    // testing
    cvdg.setValidationSet(i_vSet, true);
    double avg_error = 0;
    for (;;) {
      Vector X_t;
      double y_t;
      if (!cvdg.getNextPoint(X_t, y_t)) break;
      double loss_t = hinge_loss(w, X_t, y_t);
      if (loss_t >= 1) avg_error += 1.0;
    }
    avg_error /= cvdg.n_points();
    printf("i_vSet = %d avg_error = %f train_error = %f cvdg.n_points = %d\n", 
	   i_vSet, avg_error, train_error, cvdg.n_points());
    cv_error += avg_error;
  }
  cv_error /= cvdg.n_sets();
  fx_result_double(module, "avg_error", cv_error); 
}

int main(int argc, char** argv) {
  fx_module *root = fx_init(argc, argv, &patest_doc);  
  
  const char* filename = fx_param_str_req(root, "data");
  index_t n_laps = fx_param_int(root, "laps", 1);
  
  DatasetGenerator dg(filename, n_laps);
  //FileRowGenerator dg(filename, 1);

  const char* method = fx_param_str_req(root, "method");
  if (strcmp(method, "PA_I") == 0 || strcmp(method, "PA_II") == 0)
    fx_param_double(root, "C", 0.001);

  // Check if using kernelized version and prepare the kernel
  KernelFunction *kernel = NULL;
  if (fx_param_exists(root, "kernel")) {
    const char* kernelName = fx_param_str(root, "kernel", "linear");
    if (strcmp(kernelName, "linear")==0) kernel = new LinearKernel();
    else if (strcmp(kernelName, "poly")==0) {
      index_t order = fx_param_int(root, "order", 2);
      bool homogeneous = fx_param_int(root, "homogeneous", 0) == 1;
      kernel = new PolynomialKernel(order, homogeneous);
    }
    else if (strcmp(kernelName, "gauss")==0) {
      double sigma = fx_param_double(root, "sigma", 1.0);
      kernel = new Gaussian2Kernel(sigma);
    }
    else {
      PRINT_RESULT( fprintf(f, "data = %s kernel = %s wrong kernel name\n",
                            filename, kernelName); );
      DEBUG_ASSERT(0);
    }
  }

  // Check if doing cross validation
  if (fx_param_int(root, "CV", 0) == 1) {
    if (!kernel)
      CrossValidation(root, dg);
    else
      KernelizedCrossValidation(root, *kernel, dg);

    PRINT_RESULT(
      fprintf(f, "CV data = %s method = %s kernelized = %d "
	      "cv_error = %e avg_loss = %e " 
	      "C = %f order = %d homogeneous = %d sigma = %f\n ",
	      filename,
	      fx_param_str_req(root, "method"),
	      fx_param_exists(root, "kernel"),
	      fx_param_double(root, "avg_error", -1),
	      fx_param_double(root, "avg_loss", -1),
	      fx_param_double(root, "C", -1),
	      (int) fx_param_int(root, "order", -1),
	      (int) fx_param_int(root, "homogeneous", -1),
	      fx_param_double(root, "sigma", -1)
            );	  
      )
    fx_done(root);
    return 0;
  }
 
  // Else train on the whole training set
  // First check if linear version is required
  if (!kernel) {
    Vector weight;
    Run_PA(root, dg, weight);
    Matrix W;
    W.AliasColVector(weight);
    data::Save("weight.txt", W);    
  }
  else { // If not then do kernelized version    
    KernelizedWeight weight(dg.n_features(), *kernel);
    Run_Kernelized_PA(root, dg, weight);
    printf("n_SVs = %d\n", weight.m_lSupportVectors.size());
  }

  printf("n_points = %d n_positives = %d n_negatives = %d\n",
         dg.n_points(), dg.n_positives(), dg.n_negatives());
  PRINT_RESULT(
    fprintf(f, "data = %s method = %s kernelized = %d "
            "avg_error = %e avg_loss = %e " 
            "C = %f order = %d homogeneous = %d sigma = %f\n ",
	    filename,
            fx_param_str_req(root, "method"),
            fx_param_exists(root, "kernel"),
            fx_param_double(root, "avg_error", -1),
	    fx_param_double(root, "avg_loss", -1),
            fx_param_double(root, "C", -1),
	    (int) fx_param_int(root, "order", -1),
            (int) fx_param_int(root, "homogeneous", -1),
            fx_param_double(root, "sigma", -1)
            );	  
  )
  
  fx_done(root);
  return 0;
}

/*
void Run_PA(fx_module* module, Dataset& data, Vector& w_out) {
  w_out.Init(data.n_features()-1);
  w_out.SetZero();

  double (*update_func)(fx_module*, const Vector&, double*, double, Vector&)
    = NULL;

  const char* method = fx_param_str_req(module, "method");
  if (strcmp(method, "PA") == 0) update_func = PA_Update;
  else if (strcmp(method, "PA_I") == 0) update_func = PA_I_Update;
  else if (strcmp(method, "PA_II") == 0) update_func = PA_II_Update;
  DEBUG_ASSERT(update_func != NULL);

  double avg_error = 0;
  double avg_loss = 0;
  for (int i_p = 0; i_p < data.n_points(); i_p++) {
    double* X_t = data.point(i_p);
    double y_t = data.get(data.n_features()-1, i_p);
    Vector w_tmp;
    double loss_t = update_func(module, w_out, X_t, y_t, w_tmp);

    if (loss_t > 1) avg_error += 1.0;
    avg_loss += loss_t;

    w_out.CopyValues(w_tmp);
  }
  avg_loss /= data.n_points();
  avg_error /= data.n_points();

  fx_result_double(module, "avg_error", avg_error);
  fx_result_double(module, "avg_loss", avg_loss);
}
*/

/*
void Run_Kernelized_PA(fx_module* module, Dataset& data, KernelizedWeight& w) {
  double avg_error = 0;
  double avg_loss = 0;

  double (*update_func)(fx_module*, KernelizedWeight&, double*, double) = NULL;

  const char* method = fx_param_str_req(module, "method");
  if (strcmp(method, "PA") == 0) update_func = Kernelized_PA_Update;
  else if (strcmp(method, "PA_I") == 0) update_func = Kernelized_PA_I_Update;
  else if (strcmp(method, "PA_II") == 0) update_func = Kernelized_PA_II_Update;
  DEBUG_ASSERT(update_func != NULL);
 
  for (int i_p = 0; i_p < data.n_points(); i_p++) {
    double* X_t = data.point(i_p);
    double y_t = data.get(data.n_features()-1, i_p);
    double loss_t = update_func(module, w, X_t, y_t);

    if (loss_t > 1) avg_error += 1.0;
    avg_loss += loss_t;
  }
  avg_loss /= data.n_points();
  avg_error /= data.n_points();

  fx_result_double(module, "avg_error", avg_error);
  fx_result_double(module, "avg_loss", avg_loss);
}
*/
