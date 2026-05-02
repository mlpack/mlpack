# Signal processing

mlpack provides several signal processing techniques that allows to extract
features from stationary and non-stationary signals:

* _Decomposition_:
    - `EMD()`: adaptively decomposes a 1D signal into a set of Intrinsic Mode
 Functions (IMFs) plus a residue.
    - `EEMD():` wraps `EMD()` to output more robust IMFs by using ensemble
      approach.

* _Feature Extraction_: 
    - `MFE()`: (Mel-Frequency Energy): log-scaled energies from a bank of
   triangular filters spaced on the mel scale
    - `MFCC()`: (Mel-Frequency Cepstral Coefficients): a compact representation
   obtained by applying a Discrete Cosine Transform (DCT) to the MFE output.

## EMD 

mlpack provides Empirical Mode Decomposition (`EMD`) to process signals
for training and testing.  This can be used for signal monitoring pipelines
in nonlinear and nonstationary problems.

#### See also:

 * [Empirical Mode Decomposition on Wikipedia](https://en.wikipedia.org/wiki/Hilbert%E2%80%93Huang_transform#Empirical_mode_decomposition)
 * [EMD for nonlinear and non-stationary time series analysis](https://ui.adsabs.harvard.edu/abs/1998RSPSA.454..903H/abstract) (original EMD paper)

The `EMD()` function can be used to extract Intrinsic Mode Functions (IMFs)
from a uniformly sampled periodic signal:

#### `EMD()` Parameters

- `EMD(signal, imfs, residue, maxImfs = 10 , maxSiftIter = 50, tol = 1e-3)`
   * `signal` is a [column vector](../../matrices.md) containing the 1D signal
     data (e.g. `arma::vec`); the sequence must be uniformly sampled.

   * `imfs` is a [matrix](../../matrices.md) that will be modified to contain
     the extracted IMFs. It will have shape `N x K`, where `N` is the length
     of `signal` and `K` is the number of extracted IMFs.

   * `residue` is a column vector (of length `N`) that will be modified to
     contain the final residual signal after extracting the IMFs.

   * `maxImfs` (of type `size_t`) is the maximum number of IMFs to extract.

   * `maxSiftIter` (of type `size_t`) is the maximum sifting iterations per IMF.

   * `tol` (`double`) is the stopping tolerance used on sifting iterations.

   ***NOTES:***

   * The original signal can be reconstructed as the sum of the imfs and residue. 

   * The stopping criterion is based on the normalized mean envelope magnitude.

   * Sifting will terminate when zero-crossings and extrema are equal or
     differing by at most one. Specifically, the S-number is set to S=1.
    
   * A smaller tolerance sets a stricter stopping criterion. The algorithm will
     terminate when either `maxSiftIter` is reached or the `tol` is satisfied,
     whichever occurs first.

The figures below show the signal decomposition process with EMD:

 * (a) original signal;

 * (b) original signal with envelopes about the local minima and maxima;

 * (c) the first IMF extracted from the original signal via sifting.

<p align="center">
  <img src="../../../img/emd_visualization.svg" alt="signal with envelopes">
</p>

Example using `EMD` on a time-varying signal `S`.

```c++
const arma::uword N = 400;
const double tMin = 0.0;
const double tMax = arma::datum::pi;
arma::vec time = arma::linspace(tMin, tMax, N);

// signal = sin(20*T*(1 + 0.2*T)) + T**2 + sin(13*T)
// see figure above 
arma::vec signal =
    arma::sin( 20.0 * time % (1.0 + 0.2 * time) ) +
    arma::square(time) + arma::sin(13.0 * time);

arma::mat imfs;
arma::vec residue;

// Use up to 5 IMFs, 50 sifts per IMF, tol = 1e-6
mlpack::EMD(signal, imfs, residue, 5, 50, 5e-6);

// Print dominant frequency for the first 3 IMFs 
// to check if EMD is extracting the correct IMFs
const size_t numToShow = std::min<size_t>(3, imfs.n_cols);
const double dt = time(1) - time(0);
const double fs = 1.0 / dt;
for (size_t k = 0; k < numToShow; ++k)
{
  arma::cx_vec spectrum = arma::fft(imfs.col(k));
  // Use only the first half of the spectrum
  arma::vec mag = arma::abs(spectrum.rows(0, spectrum.n_elem / 2));
  arma::uword idx = mag.index_max();
  const double peakHz = (double) idx * fs / spectrum.n_elem;
  std::cout << "IMF " << k << " peak freq: " << peakHz << " Hz" << std::endl;
}
```

## EEMD

The `EEMD()` function wraps [`EMD()`](#emd) to produce more robust IMFs
by running EMD many times on the signal with independent white noise added
to each trial, then averaging the results. The added noise reduces mode mixing
and cancels out in the average

#### See also:

 * [Ensemble Empirical Mode Decomposition](https://perso.ens-lyon.fr/patrick.flandrin/EEMD.pdf) (original EEMD paper)
 * [EMD for nonlinear and non-stationary time series analysis](https://ui.adsabs.harvard.edu/abs/1998RSPSA.454..903H/abstract) (original EMD paper)

## `EEMD()`

- `EEMD(signal, imfs, residue, ensSize = 100, noiseStrength = 0.2, maxImfs = 10, maxSiftIter = 50, tol = 1e-3)`

   * `ensSize` (of type `size_t`) is the number of members in the ensemble
    (that is the number of `EMD()` runs to be averaged).

   * `noiseStrength` (of type `double`) is the fraction of the signal standard
   deviation used as the standard deviation of the added white noise in each
    `EMD()` run (i.e. 0.1 means 10% of signal standard deviation).

   * `signal`, `imfs`,  `residue`, `maxImfs`, `maxSiftIter`, and `tol` are
     defined in the classical `EMD()` implementation.

   ***NOTES:***

   * The original signal **cannot** be reconstructed as the sum of the imfs and
      residue, as in `EMD()`.

   * Number of extracted IMFs will be the minimum number of IMFs extracted by
     by `EMD()` across all `ensSize` runs (`<=maxImfs`).

   * EEMD may produce low-energy leading IMFs due to injected noise and ensemble
   averaging. Depending on the application, users may want to discard negligible
   IMFs in post-processing (e.g., using an energy-fraction threshold).

   * The number of returned IMFs is bounded between `0` and `maxImfs`.
     The algorithm returns only as many IMFs as can actually be extracted from
     the input signal.

Example using `EEMD` on a time-varying signal `S` (shown in above EMD figure).

```c++
const arma::uword N = 3000;
const double tMin = 0.0;
const double tMax = arma::datum::pi;
arma::vec time = arma::linspace(tMin, tMax, N);

// signal = sin(20*T*(1 + 0.2*T)) + T**2 + sin(13*T)
// see figure above in the EMD documentation
arma::vec signal =
    arma::sin( 20.0 * time % (1.0 + 0.2 * time) ) +
    arma::square(time) + arma::sin(13.0 * time);

arma::mat imfs;
arma::vec residue;

// Use 100 ensemble members, 0.15 noise strength, 10 IMFs, 50 sifts per IMF, tol = 1e-2
mlpack::EEMD(signal, imfs, residue, 100, 0.15, 10, 50, 1e-2);
```

## MFE

mlpack provides the `MFE()` functions to extract standard audio
features from raw PCM data loaded with [`Load()`](../../load_save.md).  These features are
used as input to machine learning models for speech recognition, speaker identification,
keyword spotting

#### See also:

 * [Audio data loading and saving](../../load_save.md#audio-data)
 * [Mel-frequency cepstrum](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)
 * [MFCC tutorial (Practical Cryptography)](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/)
 * [The cepstrum, mel-cepstrum, and MFCCs (Aalto University)](https://speechprocessingbook.aalto.fi/Representations/Melcepstrum.html)
 * [Mel-Spectrogram and MFCCs | Lecture 72 (Part 1) | Applied Deep Learning](https://github.com/maziarraissi/Applied-Deep-Learning/blob/main/06%20-%20Speech%20%26%20Music/01%20-%20Recognition.pdf)

## `MFE()`

 - `MFE(signals, output, sampleRate, numMelFilters=40, windowLength=25,
   windowStep=10, nFFT=0, lowFreq=0.0, highFreq=0.0)`
   * Extract log-mel filterbank energies.

   * `signals` input matrix contains a columm wise represented signals.
   
   * `mfe` output matrix of shape `(numMelFilters x numWindows)`

   * `sampleRate` provided by `AudioOptions` after loading audio data.

   * If the audio signal was loaded as an integral type, you can either
     change the loading code to load directly into a floating-point type matrix, or
     use conv_to to convert to a floating-point type, then scale the range to [-1, 1]

   * If the audio signal has multiple channels, the channels must be
     de-interleaved before calling `MFE()` with each column representing a separate
     channel.  This preserves spatial information (e.g., which sounds come from the
     left vs. right).  If spatial information is not needed, the channels can be
     mixed down to mono before processing (e.g., `mono = (left + right) / 2`).
     Both cases are demonstrated in the examples below.

### Functions Parameters:

|     **name**     |  **type**   |   **default**  | **description**                                         |
|------------------|-------------|----------------|---------------------------------------------------------|
| `signals`        | `arma::mat` or other floating-point matrix | _(n/a)_ | raw pcm audio samples.                                         |
| `mfe`            | `arma::mat` or other floating point matrix | _(n/a)_ | output matrix of shape `(numMelFilters x numWindows)`.         |
| `sampleRate`     | `size_t`    | _(n/a)_ | sample rate of the audio in hz (e.g. `16000`, `44100`).        |
| `numMelFilters`  | `size_t`    | `40`    | number of mel-spaced triangular filters. Typical range (`20` to `100`) |
| `windowLength`   | `float`     | `25.0`  | window length in milliseconds.  Typical range (`20` to `40`)           |
| `windowStep`     | `float`     | `10.0`  | window hop (step) in milliseconds. Typical range (`5` to `20`)         |
| `nFFT`           | `size_t`    | `0`     | fft size; `0` means the number of points fed to fft is chosen automatically using the next power of 2 >= of the window length. Typical range (`256` to `4096`) |
| `lowFreq`        | `float`     | `0.0`   | low frequency bound for the mel filterbank in hz. Typical range (`0` to `300`) |
| `highFreq`       | `float`     | `0.0`   | high frequency bound in hz; `0` means `sampleRate / 2`. Typical range (`4000` to sampleRate / 2) |

---

Apply MFE filter with default parameters on voice signals:

```c++
// See https://datasets.mlpack.org/sine.wav
arma::mat signal;
mlpack::AudioOptions opts = mlpack::Fatal + mlpack::WAV;
mlpack::Load("sine.wav", signal, opts);

arma::mat mfe;
mlpack::MFE(signal, mfe, opts.SampleRate());

std::cout << "MFE shape: " << mfe.n_rows << " x " << mfe.n_cols << std::endl;

```

Specifying a custom number of mel filters and frequency range:

```c++
// See https://datasets.mlpack.org/sine.wav
arma::fmat signal;
mlpack::AudioOptions opts = mlpack::Fatal + mlpack::WAV;
mlpack::Load("sine.wav", signal, opts);

arma::fmat mfe;
// 80 mel filters, default window size, frequency range 300–8000 Hz.
mlpack::MFE(signal, mfe, opts.SampleRate(), 80, 25.0, 10.0, 0, 300.0,
    8000.0);
```
---

## MFCC

mlpack provides the `MFCC()` functions to extract standard audio
features from raw PCM data loaded with [`Load()`](../../load_save.md).  Similar to MFE, MFCC
output can be used as input to machine learning models. Note that, since MFCC
coefficient are decorrelated, it can be combined with several distance-based 
machine learning algorithms (e.g., KNN, KMeans) or probabilitic algorithms
(e.g., GMM, HMM).

MFCC is a superset of MFE, the MFE pipeline produces the first step as an
intermidiate representation, and a single additional DCT step to generate MFCC
coefficients.

## `MFCC()`

 - `MFCC(signals, output, sampleRate, numCoeffs, numMelFilters, windowLength,
   windowStep, nFFT, lowFreq, highFreq)`
   * Extract MFCC with different number coefficients dependings on the user specified
 
   * `signals` input matrix contains a columm wise represented signals.
   
   * `mfcc` output matrix of shape `(numCoeffs x numWindows)`

   * `sampleRate` provided by `AudioOptions` after loading audio data.
    parameters.

   * if the audio signal was loaded as an integral type, you can either
     change the loading code to load directly into a floating-point type matrix, or
     use conv_to to convert to a floating-point type, then scale the range to [-1, 1]

   * If the audio signal has multiple channels, the channels must be
     de-interleaved before calling `MFCC()` with each column representing a separate
     channel.  This preserves spatial information (e.g., which sounds come from the
     left vs. right).  If spatial information is not needed, the channels can be
     mixed down to mono before processing (e.g., `mono = (left + right) / 2`).
     Both cases are demonstrated in the examples below.

   * `numCoeffs` must be less than or equal to `numMelFilters`.  The
     DCT compresses `numMelFilters` log-mel energies down to `numCoeffs` cepstral
     coefficients — it cannot produce more coefficients than there are input
     values.

### Functions Parameters:

|     **name**     |  **type**   |   **default**  | **description**                                  |
|------------------|-------------|----------------|--------------------------------------------------|
| `signals`        | `arma::mat` or other floating-point matrix | _(n/a)_ | raw pcm audio samples.                                  |
| `mfcc`           | `arma::mat` or other floating point matrix | _(n/a)_ | output matrix of shape `(numMelFilters x numWindows)`.  |
| `sampleRate`     | `size_t`    | _(n/a)_ | sample rate of the audio in hz (e.g. `16000`, `44100`). |
| `numCoeff`       | `size_t`    | _(n/a)_ | Number of cepstral coefficients.                        |
| `numMelFilters`  | `size_t`    | `40`    | number of mel-spaced triangular filters.  Typical range (`20` to `100`)    |
| `windowLength`   | `float`     | `25.0`  | window length in milliseconds. Typical range (`20` to `40`)                |
| `windowStep`     | `float`     | `10.0`  | window hop (step) in milliseconds.  Typical range (`5` to `20`)            |
| `nFft`           | `size_t`    | `0`     | fft size; `0` means the number of points fed to fft is chosen automatically using the next power of 2 >= of the window length. Typical range (`256` to `4096`) |
| `lowFreq`        | `float`     | `0.0`   | low frequency bound for the mel filterbank in hz. Typical range (`0` to `300`)      |
| `highFreq`       | `float`     | `0.0`   | high frequency bound in hz; `0` means `sampleRate / 2`. Typical range (`4000` to sampleRate / 2) |

---

Extract 13 MFCCs from a WAV file.

```c++
// See https://datasets.mlpack.org/sine.wav
arma::mat signal;
mlpack::AudioOptions opts = mlpack::Fatal + mlpack::WAV;
mlpack::Load("sine.wav", signal, opts);

arma::mat mfcc;
mlpack::MFCC(signal, mfcc, opts.SampleRate());

// mfcc has a shape of 13 x numWindows.
std::cout << "MFCC shape: " << mfcc.n_rows << " x " << mfcc.n_cols
    << std::endl;
```

Extract 20 MFCCs with 80 mel filters from an MP3 file:

```c++
// See https://datasets.mlpack.org/fifths.mp3
arma::fmat signal;
mlpack::AudioOptions opts = mlpack::Fatal + mlpack::MP3;
mlpack::Load("fifths.mp3", signal, opts);

arma::fmat mfcc;
mlpack::MFCC(signal, mfcc, opts.SampleRate(), 20, 80);
```

MFCC with custom parameters:

```c++
// See https://datasets.mlpack.org/sine.wav
arma::fmat signal;
mlpack::AudioOptions opts = mlpack::Fatal + mlpack::WAV;
mlpack::Load("sine.wav", signal, opts);

arma::fmat mfcc;
mlpack::MFCC(signal, mfcc, opts.SampleRate(), 13, 26, 25.0, 10.0, 512, 300.0,
    3400.0);
```
