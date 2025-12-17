# `EMD`

Empirical Mode Decomposition (EMD) adaptively decomposes a 1D signal into a
small set of Intrinsic Mode Functions (IMFs) plus a residue. This is often used
for vibration/sensor pipelines as a time-domain alternative to FFT.

## Inputs

- `signal` (`arma::Col<eT>`): uniformly sampled 1D signal (any Armadillo
  column type / scalar `eT`).
- `maxImfs` (`size_t`, default `10`): maximum number of IMFs to extract.
- `maxSiftIter` (`size_t`, default `10`): maximum sifting iterations per IMF.
- `tol` (`double`, default `1e-3`): sifting stopping tolerance (relative L2
  change per iteration).

## Outputs

- `imfs` (`arma::Mat<eT>`): shape `N x K`, each column an IMF, `K <= maxImfs`.
- `residue` (`arma::Col<eT>`): final residual signal after removing all IMFs.

## Reconstruction

The original signal can be reconstructed as

`signal ~ Σ_k imfs.col(k) + residue`

## Example

```c++
// Example: simple sine + trend.
arma::vec t = arma::linspace(0.0, 1.0, 1000);
arma::vec signal = arma::sin(2.0 * arma::datum::pi * 10.0 * t) + 0.5 * t;

arma::mat imfs;
arma::vec residue;

// Use defaults: up to 10 IMFs, 10 sifts per IMF, tol = 1e-3
mlpack::EMD(signal, imfs, residue);

  // Or use custom settings: 7 IMFs, 6 sifts/IMF, tighter tolerance.
EMD(signal, imfs, residue, 7 /* maxImfs */, 6 /* maxSiftIter */, 5e-4);
return 0;
```