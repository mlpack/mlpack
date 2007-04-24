#include "svm.h"

void PCA(Dataset* dataset) {
  Matrix u;
  Matrix vt;
  Vector s;
  Vector values;
  
  values.Init(dataset->n_points());
  for (index_t i = 0; i < values.length(); i++) {
    double *v = &dataset->matrix().ref(dataset->n_features() - 1, i);
    values[i] = *v;
    *v = 0;
  }
  
  la::SVDInit(dataset->matrix(), &s, &u, &vt);
  
  /* Normalize singular values to length 1. */
  la::Scale(1.0 / s[0], &s);
  
  la::ScaleRows(s, &vt);
  
  dataset->matrix().CopyValues(vt);

  for (index_t i = 0; i < values.length(); i++) {
    double *v = &dataset->matrix().ref(dataset->n_features() - 1, i);
      *v = values[i];
  }
  
  //dataset->matrix().PrintDebug("m");
}

int main(int argc, char *argv[]) {
  fx_init(argc, argv);
  
  Dataset dataset;
  
  if (fx_param_exists(NULL, "data")) {
    // if a data file is specified, use it.
    if (!PASSED(dataset.InitFromFile(fx_param_str_req(NULL, "data")))) {
      fprintf(stderr, "Couldn't open the data file.\n");
      return 1;
    }
  } else {
    // create an artificial dataset and save it to "m.csv"
    
    Matrix m;
    index_t n = fx_param_int(NULL, "n", 30);
    double offset = fx_param_double(NULL, "offset", 0.0);
    double range = fx_param_double(NULL, "range", 1.0);
    double slope = fx_param_double(NULL, "slope", 1.0);
    double margin = fx_param_double(NULL, "margin", 1.0);
    double var = fx_param_double(NULL, "var", 1.0);
    double intercept = fx_param_double(NULL, "intercept", 0.0);
    
    // 3 dimensional dataset, size n
    m.Init(3, n);
    
    for (index_t i = 0; i < n; i += 2) {
      double x;
      double y;
      
      x = (rand() * range / RAND_MAX) + offset;
      y = margin / 2 + (rand() * var / RAND_MAX);
      m.set(0, i, x);
      m.set(1, i, x*slope + y + intercept);
      m.set(2, i, 0);
      
      x = (rand() * range / RAND_MAX) + offset;
      y = margin / 2 + (rand() * var / RAND_MAX);
      m.set(0, i+1, x);
      m.set(1, i+1, x*slope - y + intercept);
      m.set(2, i+1, 1);
    }
    data::Save("m.csv", m);
    dataset.OwnMatrix(&m);
  }
  
  if (fx_param_exists(NULL, "pca")) {
    fprintf(stderr, "Doing PCA\n");
    PCA(&dataset);
  } else {
    fprintf(stderr, "Skipping PCA\n");
  }
  
  SimpleCrossValidator< SVM<SVMRBFKernel> > cross_validator;
  cross_validator.Init(&dataset, 2, 2, fx_root, "svm");
  cross_validator.Run(true);
  
  fx_done();
}

