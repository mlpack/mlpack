# `EMD`

Empirical Mode Decomposition (EMD) adaptively decomposes a 1D signal into a
small set of Intrinsic Mode Functions (IMFs) plus a residue. This is often used
for vibration/sensor pipelines as a time-domain alternative to FFT.

## Inputs

- `signal` (`arma::Col<eT>`): uniformly sampled 1D signal (any Armadillo
  column type / scalar `eT`).
- `maxImfs` (`size_t`, default `10`): maximum number of IMFs to extract.
- `maxSiftIter` (`size_t`, default `10`): maximum sifting iterations per IMF.
- `tol` (`double`, default `1e-3`): sifting stopping tolerance on the mean
  envelope ratio (see notes below).

## Outputs

- `imfs` (`arma::Mat<eT>`): shape `N x K`, each column an IMF, `K <= maxImfs`.
- `residue` (`arma::Col<eT>`): final residual signal after removing all IMFs.

## Reconstruction

The original signal can be reconstructed as

`signal ~ Σ_k imfs.col(k) + residue`

## Example

```c++

arma::vec t = arma::linspace<arma::vec>(0.0, 1.0, 1000);
// Example: pick 5 frequencies to construct a non-stationary signal
const double A = 5.0;   // Hz
const double B = 15.0;  // Hz
const double C = 25.0;  // Hz
const double D = 35.0;  // Hz
const double E = 55.0;  // Hz

// Time shifts so each component dominates a different part of the signal
const double tA = 0.00, tB = 0.15, tC = 0.35, tD = 0.55, tE = 0.75;
arma::vec wA = arma::clamp(1.0 - 4.0 * arma::abs(t - 0.10), 0.0, 1.0);
arma::vec wB = arma::clamp(1.0 - 4.0 * arma::abs(t - 0.30), 0.0, 1.0);
arma::vec wC = arma::clamp(1.0 - 4.0 * arma::abs(t - 0.50), 0.0, 1.0);
arma::vec wD = arma::clamp(1.0 - 4.0 * arma::abs(t - 0.70), 0.0, 1.0);
arma::vec wE = arma::clamp(1.0 - 4.0 * arma::abs(t - 0.90), 0.0, 1.0);

arma::vec signal =
  wA % arma::sin(2.0 * arma::datum::pi * A * (t - tA)) +
  wB % arma::sin(2.0 * arma::datum::pi * B * (t - tB)) +
  wC % arma::sin(2.0 * arma::datum::pi * C * (t - tC)) +
  wD % arma::sin(2.0 * arma::datum::pi * D * (t - tD)) +
  wE % arma::sin(2.0 * arma::datum::pi * E * (t - tE)) +
  0.1 * t; // small trend

arma::mat imfs;
arma::vec residue;

// Use up to 10 IMFs, 30 sifts per IMF, tol = 1e-3
mlpack::EMD(signal, imfs, residue);

// Print dominant frequency for the first five IMFs 
//check if EMD is extracting the correct IMFs
const size_t numToShow = std::min<size_t>(5, imfs.n_cols);
const double dt = t(1) - t(0);
const double fs = 1.0 / dt;
for (size_t k = 0; k < numToShow; ++k)
{
  arma::cx_vec spectrum = arma::fft(imfs.col(k));
  arma::vec mag = arma::abs(spectrum.rows(0, spectrum.n_elem / 2));
  arma::uword idx = 0;
  mag.max(idx);
  const double peakHz = (double) idx * fs / spectrum.n_elem;
  std::cout << "IMF " << k << " peak freq ~ " << peakHz << " Hz" << std::endl;
}

  // Or use custom settings: 15 IMFs, 15 sifts/IMF, tighter tolerance.
// EMD(signal, imfs, residue, 15 /* maxImfs */, 15 /* maxSiftIter */, 5e-4);
return 0;
```