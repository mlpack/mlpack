#include "mex.h"

#include <mlpack/core.hpp>
#include <mlpack/core/kernels/linear_kernel.hpp>
#include <mlpack/core/kernels/gaussian_kernel.hpp>
#include <mlpack/core/kernels/hyperbolic_tangent_kernel.hpp>
#include <mlpack/core/kernels/laplacian_kernel.hpp>
#include <mlpack/core/kernels/polynomial_kernel.hpp>
#include <mlpack/core/kernels/cosine_distance.hpp>

#include <mlpack/methods/kernel_pca/kernel_pca.hpp>

using namespace mlpack;
using namespace mlpack::kpca;
using namespace mlpack::kernel;
using namespace std;
using namespace arma;

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
  // argument checks
  if (nrhs != 8)
  {
    mexErrMsgTxt("Expecting eight arguments.");
  }

  if (nlhs != 1)
  {
    mexErrMsgTxt("Output required.");
  }

  // Load input dataset.
  if (mxDOUBLE_CLASS != mxGetClassID(prhs[0]))
    mexErrMsgTxt("Input dataset must have type mxDOUBLE_CLASS.");

  mat dataset(mxGetM(prhs[0]), mxGetN(prhs[0]));
  double * values = mxGetPr(prhs[0]);
  for (int i=0, num=mxGetNumberOfElements(prhs[0]); i<num; ++i)
    dataset(i) = values[i];

  // Get the new dimensionality, if it is necessary.
  size_t newDim = dataset.n_rows;
  const int argNewDim = (int) mxGetScalar(prhs[2]);
  if (argNewDim != 0)
  {
    newDim = argNewDim;

    if (newDim > dataset.n_rows)
    {
      stringstream ss;
      ss << "New dimensionality (" << newDim
          << ") cannot be greater than existing dimensionality ("
          << dataset.n_rows << ")!";
      mexErrMsgTxt(ss.str().c_str());
    }
  }

  // Get the kernel type and make sure it is valid.
  if (mxCHAR_CLASS != mxGetClassID(prhs[1]))
  {
    mexErrMsgTxt("Kernel input must have type mxCHAR_CLASS.");
  }
  int bufLength = mxGetNumberOfElements(prhs[1]) + 1;
  char * buf;
  buf = (char *) mxCalloc(bufLength, sizeof(char));
  mxGetString(prhs[1], buf, bufLength);
  string kernelType(buf);
  mxFree(buf);

  // scale parameter
  const bool scaleData = (mxGetScalar(prhs[3]) == 1.0);

  if (kernelType == "linear")
  {
    KernelPCA<LinearKernel> kpca(LinearKernel(), scaleData);
    kpca.Apply(dataset, newDim);
  }
  else if (kernelType == "gaussian")
  {
    const double bandwidth = mxGetScalar(prhs[3]);

    GaussianKernel kernel(bandwidth);
    KernelPCA<GaussianKernel> kpca(kernel, scaleData);
    kpca.Apply(dataset, newDim);
  }
  else if (kernelType == "polynomial")
  {
    const double degree = mxGetScalar(prhs[4]);
    const double offset = mxGetScalar(prhs[5]);

    PolynomialKernel kernel(offset, degree);
    KernelPCA<PolynomialKernel> kpca(kernel, scaleData);
    kpca.Apply(dataset, newDim);
  }
  else if (kernelType == "hyptan")
  {
    const double scale = mxGetScalar(prhs[6]);
    const double offset = mxGetScalar(prhs[5]);

    HyperbolicTangentKernel kernel(scale, offset);
    KernelPCA<HyperbolicTangentKernel> kpca(kernel, scaleData);
    kpca.Apply(dataset, newDim);
  }
  else if (kernelType == "laplacian")
  {
    const double bandwidth = mxGetScalar(prhs[7]);

    LaplacianKernel kernel(bandwidth);
    KernelPCA<LaplacianKernel> kpca(kernel, scaleData);
    kpca.Apply(dataset, newDim);
  }
  else if (kernelType == "cosine")
  {
    KernelPCA<CosineDistance> kpca(CosineDistance(), scaleData);
    kpca.Apply(dataset, newDim);
  }
  else
  {
    // Invalid kernel type.
    stringstream ss;
    ss << "Invalid kernel type ('" << kernelType << "'); valid choices "
        << "are 'linear', 'gaussian', 'polynomial', 'hyptan', 'laplacian', and "
        << "'cosine'.";
    mexErrMsgTxt(ss.str().c_str());
  }

  // Now returning results to matlab
  plhs[0] = mxCreateDoubleMatrix(dataset.n_rows, dataset.n_cols, mxREAL);
  values = mxGetPr(plhs[0]);
  for (int i = 0; i < dataset.n_rows * dataset.n_cols; ++i)
  {
    values[i] = dataset(i);
  }

}
