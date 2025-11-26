
 # `EMD`

 ML pipelines trained on data from vibration sensors require advanced data 
 preprocessing techniques that may be as expensive as the model training. 
 Empirical Mode Decomposition is used heavily in the signal processing 
 applications as an alternative to FFT (which can be used through armadillo).

## Data

Input 
`signal` (`arma::Col<eT>`)

- Column vector 
- Any armadillo compatible scaler type `<eT>` 
- Assumed to be uniformly sampled in time (does not consider a time vector)

`maxImfs` (`size_t`, default `10`)

- number of IMFs to extract

`maxSiftIter` (default `10`)

- Max number of sifting iterations used to extract a single IMF

`tol` (default `1e-3`)

- tolerance for the IMF sifting step 
- controls how much the IMF changes per sift iteration

Output 
`imfs` (`arma::` Mat<eT>)

- `N x K` matrix, where each col is one IMF and `K` is $\leq$ `maxImfs`

`residue` (`arma::Col<eT>`)
- Residual signal left over from removing IMFS

Reconstructing the original signal using the decomposed elements looks like:

$\mathrm{signal} \approx \sum_{k=1}^{K} \mathrm{imfs.col}(k) + \mathrm{residue} $

 ## Example
 Example usage:
 ```c++
#include <mlpack/methods/emd/emd.hpp>

using namespace mlpack;
using namespace mlpack::emd;

int main()
{
  // Example: simple sine + trend.
  arma::vec t = arma::linspace(0.0, 1.0, 1000);
  arma::vec signal = arma::sin(2.0 * arma::datum::pi * 10.0 * t) + 0.5 * t;

  arma::mat imfs;
  arma::vec residue;

  // Use defaults: up to 10 IMFs, 10 sifts per IMF, tol = 1e-3
  Emd(signal, imfs, residue);

    // or use custom : 7 IMFs (maxImfs), 6 maxSiftIter per IMF, tol = 5e-4
  Emd(signal, imfs, residue, 7, 6, 5e-4)
  return 0;
}
 ```




