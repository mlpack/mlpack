#ifndef __MLPACK_METHODS_NNSVM_NNSVM_IMPL_HPP
#define __MLPACK_METHODS_NNSVM_NNSVM_IMPL_HPP

namespace mlpack {
namespace nnsvm {

/**
* NNSVM initialization
*
* @param: labeled training set
* @param: number of classes (different labels) in the data set
* @param: module name
*/
template<typename TKernel>
void NNSVM<TKernel>::Init(const arma::mat& dataset, size_t n_classes)
{
  Init(dataset, n_classes, 10, dataset.n_rows, 1.0e-6, 1000);
}
template<typename TKernel>
void NNSVM<TKernel>::Init(const arma::mat& dataset, size_t n_classes, size_t c, size_t b, double eps, size_t max_iter)
{
  // c; default:10
  param_.c_ = c;
  // budget parameter, controls # of support vectors; default: # of data samples
  if(!mlpack::CLI::HasParam("nnsvm/b"))
    mlpack::CLI::GetParam<double>("nnsvm/b") = dataset.n_rows;

  param_.b_ = b;
  // tolerance: eps, default: 1.0e-6
  param_.eps_ = eps;
  //max iterations: max_iter, default: 1000
  param_.max_iter_ = max_iter;
  fprintf(stderr, "c=%f, eps=%g, max_iter=%zu \n", param_.c_, param_.eps_, param_.max_iter_);
}

/**
* Initialization(data dependent) and training for NNSVM Classifier
*
* @param: labeled training set
* @param: number of classes (different labels) in the training set
* @param: module name
*/
template<typename TKernel>
void NNSVM<TKernel>::InitTrain(
    const arma::mat& dataset, size_t n_classes)
{
  InitTrain(dataset, n_classes, 10, dataset.n_rows, 1.0e-6, 1000);
}
template<typename TKernel>
void NNSVM<TKernel>::InitTrain(
    const arma::mat& dataset, size_t n_classes, size_t c, size_t b, double eps, size_t max_iter)
{
  std::cerr << "made it to " << __LINE__ << " in "__FILE__"\n";
  Init(dataset, n_classes, c, b, eps, max_iter);
  /* # of features = # of rows in data matrix - 1, as last row is for labels*/
  num_features_ = dataset.n_rows - 1;
  Log::Assert(n_classes == 2, "SVM is only a binary classifier");
  CLI::GetParam<std::string>("kernel_type") = typeid(TKernel).name();

  /* Initialize parameters c_, budget_, eps_, max_iter_, VTA_, alpha_, error_, thresh_ */
  NNSMO<Kernel> nnsmo;
  nnsmo.Init(dataset, param_.c_, param_.b_, param_.eps_, param_.max_iter_);

  /* 2-classes NNSVM training using NNSMO */
  Timers::StartTimer("nnsvm/nnsvm_train");
  nnsmo.Train();
  Timers::StopTimer("nnsvm/nnsvm_train");

  /* Get the trained bi-class model */
  nnsmo.GetNNSVM(support_vectors_, model_.sv_coef_, model_.w_);
  std::cerr << "the NUMBER of elements in sv_coef_ is " << model_.sv_coef_.n_elem << "\n";
  mlpack::Log::Assert(model_.sv_coef_.n_elem != 0);
  model_.num_sv_ = support_vectors_.n_cols;
  model_.thresh_ = nnsmo.threshold();
  //DEBUG_ONLY(fprintf(stderr, "THRESHOLD: %f\n", model_.thresh_));

  /* Save models to file "nnsvm_model" */
  SaveModel("nnsvm_model"); // TODO: param_req
}

/**
* Save the NNSVM model to a text file
*
* @param: name of the model file
*/
template<typename TKernel>
void NNSVM<TKernel>::SaveModel(std::string modelfilename)
{
  // TODO: Why do we do this? 
  FILE *fp = fopen(modelfilename.c_str(), "w");
  if (fp == NULL)
  {
    fprintf(stderr, "Cannot save trained model to file!");
    return;
  }

  fprintf(fp, "svm_type svm_c\n"); // TODO: svm-mu, svm-regression...
  // save kernel parameters
 // param_.kernel_.SaveParam(fp);
  fprintf(fp, "total_num_sv %zu\n", model_.num_sv_);
  fprintf(fp, "threshold %g\n", model_.thresh_);
  fprintf(fp, "weights");
  size_t len = model_.w_.n_elem;
  for(size_t s = 0; s < len; s++)
    fprintf(fp, " %f", model_.w_[s]);
  fprintf(fp, "\nsvs\n");
  for(size_t i=0; i < model_.num_sv_; i++)
  {
     fprintf(fp, "%f ", model_.sv_coef_[i]);
     for(size_t s=0; s < num_features_; s++)
     {
       fprintf(fp, "%f ", support_vectors_(s, i));
     }
     fprintf(fp, "\n");
  }
  fclose(fp);
}

/**
* Load NNSVM model file
*
* @param: name of the model file
*/
// TODO: use XML
template<typename TKernel>
void NNSVM<TKernel>::LoadModel(arma::mat& testset, std::string modelfilename)
{
  /* Init */
  //fprintf(stderr, "modelfilename= %s\n", modelfilename.c_str());
  num_features_ = testset.n_cols - 1;

  model_.w_.set_size(num_features_);
  /* load model file */
  FILE *fp = fopen(modelfilename.c_str(), "r");
  if (fp == NULL)
  {
    fprintf(stderr, "Cannot open NNSVM model file!");
    return;
  }
  char cmd[80];
  size_t i, j;
  double temp_f;
  while (1)
  {
    fscanf(fp, "%80s", cmd);
    if(strcmp(cmd,"svm_type") == 0)
    {
      fscanf(fp, "%80s", cmd);
      if(strcmp(cmd, "svm_c") == 0)
      {
        fprintf(stderr, "SVM_C\n");
      }
    }
    else if (strcmp(cmd, "total_num_sv") == 0)
    {
      fscanf(fp, "%zu", &model_.num_sv_);
    }
    else if (strcmp(cmd, "threshold") == 0)
    {
      fscanf(fp, "%lf", &model_.thresh_);
    }
    else if (strcmp(cmd, "weights")==0)
    {
      for (size_t s= 0; s < num_features_; s++)
      {
        fscanf(fp, "%lf", &temp_f);
        model_.w_[s] = temp_f;
      }
      break;
    }
  }
  support_vectors_.set_size(num_features_, model_.num_sv_);
  model_.sv_coef_.set_size(model_.num_sv_);

  while (1)
  {
    fscanf(fp, "%80s", cmd);
    if (strcmp(cmd, "svs") == 0)
    {
      for (i = 0; i < model_.num_sv_; i++)
      {
        fscanf(fp, "%lf", &temp_f);
        model_.sv_coef_[i] = temp_f;
        for (j = 0; j < num_features_; j++)
        {
          fscanf(fp, "%lf", &temp_f);
          support_vectors_(j, i) = temp_f;
        }
      }
      break;
    }
  }
  fclose(fp);
}

/**
* NNSVM classification for one testing vector
*
* @param: testing vector
*
* @return: a label (integer)
*/

template<typename TKernel>
size_t NNSVM<TKernel>::Classify(const arma::vec& datum)
{
  double summation = dot(model_.w_, datum);

  return (summation - model_.thresh_ > 0.0) ? 1 : 0;

  return 0;
}

/**
* Online batch classification for multiple testing vectors. No need to load model file,
* since models are already in RAM.
*
* Note: for test set, if no true test labels provided, just put some dummy labels
* (e.g. all -1) in the last row of testset
*
* @param: testing set
* @param: file name of the testing data
*/
template<typename TKernel>
void NNSVM<TKernel>::BatchClassify(arma::mat& testset, std::string testlablefilename)
{
  FILE *fp = fopen(testlablefilename.c_str(), "w");
  if (fp == NULL)
  {
    mlpack::Log::Fatal << "Cannot save test labels to file!" << std::endl;
    return;
  }
  num_features_ = testset.n_cols - 1;
  for (size_t i = 0; i < testset.n_rows; i++)
  {
    arma::vec testvec(num_features_);
    for(size_t j = 0; j < num_features_; j++)
    {
      testvec[j] = testset(j, i);
    }
    size_t testlabel = Classify(testvec);
    fprintf(fp, "%zu\n", testlabel);
  }
  fclose(fp);
}

/**
* Load models from a file, and perform offline batch classification for multiple testing vectors
*
* @param: testing set
* @param: name of the model file
* @param: name of the file to store classified labels
*/
template<typename TKernel>
void NNSVM<TKernel>::LoadModelBatchClassify(arma::mat& testset, std::string modelfilename, std::string testlabelfilename)
{
  LoadModel(testset, modelfilename);
  BatchClassify(testset, testlabelfilename);
}

}; // namespace nnsvm
}; // namespace mlpack

#endif // __MLPACK_METHODS_NNSVM_NNSVM_IMPL_HPP
