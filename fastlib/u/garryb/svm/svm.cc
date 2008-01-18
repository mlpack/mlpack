#include "svm.h"

void DoSvmNormalize(Dataset* dataset) {
  Matrix m;
  Vector sums;

  m.Init(dataset->n_features()-1, dataset->n_points());
  sums.Init(dataset->n_features() - 1);
  sums.SetZero();

  for (index_t i = 0; i < dataset->n_points(); i++) {
    Vector s;
    Vector d;
    dataset->matrix().MakeColumnSubvector(i, 0, dataset->n_features()-1, &s);
    m.MakeColumnVector(i, &d);
    d.CopyValues(s);
    la::AddTo(s, &sums);
  }
  
  la::Scale(-1.0 / dataset->n_points(), &sums);
  for (index_t i = 0; i < dataset->n_points(); i++) {
    Vector d;
    m.MakeColumnVector(i, &d);
    la::AddTo(sums, &d);
  }
  
  Matrix cov;


  la::MulTransBInit(m, m, &cov);

  Vector d;
  Matrix u; // eigenvectors
  Matrix ui; // the inverse of eigenvectors

  //cov.PrintDebug("cov");
  la::EigenvectorsInit(cov, &d, &u);
  la::TransposeInit(u, &ui);

  for (index_t i = 0; i < d.length(); i++) {
    d[i] = 1.0 / sqrt(d[i] / (dataset->n_points() - 1));
  }

  la::ScaleRows(d, &ui);

  Matrix cov_inv_half;
  la::MulInit(u, ui, &cov_inv_half);

  Matrix final;
  la::MulInit(cov_inv_half, m, &final);

  for (index_t i = 0; i < dataset->n_points(); i++) {
    Vector s;
    Vector d;
    dataset->matrix().MakeColumnSubvector(i, 0, dataset->n_features()-1, &d);
    final.MakeColumnVector(i, &s);
    d.CopyValues(s);
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
  
  if (fx_param_bool(NULL, "normalize", 1)) {
    fprintf(stderr, "Normalizing\n");
    DoSvmNormalize(&dataset);
  } else {
    fprintf(stderr, "Skipping normalize\n");
  }
  
  if (fx_param_bool(NULL, "save", 0)) {
    fx_default_param(NULL, "kfold/save", "1");
    dataset.WriteCsv("normalized.csv");
  }
  
  SimpleCrossValidator< SVM<SVMRBFKernel> > cross_validator;
  // k_cv: number of cross-validation folds
  cross_validator.Init(&dataset, 2,fx_param_int_req(NULL,"k_cv"), fx_root, "svm");
  cross_validator.Run(true);
  
  cross_validator.confusion_matrix().PrintDebug("confusion matrix");
  
  fx_done();
}

