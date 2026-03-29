# Signal preprocessing

<!-- 
    EMD goes here, once it is merged, I will merge the introduction of both
    and then refer to the Audio features section.
-->

## Audio Feature Extraction

mlpack provides the `MFE()` and `MFCC()` functions to extract standard audio
features from raw PCM data loaded with [`Load()`](#load).  These features are
used as input to machine learning models for speech recognition, speaker identification,
keyword spotting, and other audio related piplines.

 - **MFE** (Mel-Frequency Energy): log-scaled energies from a bank of
   triangular filters spaced on the mel scale.
 - **MFCC** (Mel-Frequency Cepstral Coefficients): a compact representation
   obtained by applying a Discrete Cosine Transform (DCT) to the MFE output.

MFCC is a superset of MFE, the MFE pipeline produces the first step as an
intermidiate representation, and a single additional DCT step to generate MFCC
coefficients.

#### See also:

 * [Audio data loading and saving](#audio-data)
 * [Mel-frequency cepstrum](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)
 * [MFCC tutorial (Practical Cryptography)](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/)
 * [The cepstrum, mel-cepstrum, and MFCCs (Aalto University)](https://speechprocessingbook.aalto.fi/Representations/Melcepstrum.html)

<!-- 
    @rcurtin, I have the links for a couple of very good youtube videos that
    helped explaining all the steps for the algorithm, if you would like, we
    can add them here as well.
-->

## `MFE()` and `MFCC()`

 - `MFE(signal, sampleRate, output)`
   * Extract log-mel filterbank energies from `signal` using default parameters.

 - `MFE(signal, sampleRate, output, numMelFilters, windowLength, windowStep,
   nFFT, lowFreq, highFreq, preEmphCoeff)`
   * Extract log-mel filterbank energies depending on the user specified
     parameters .

 - `MFCC(signal, sampleRate, output)`
   * Extract 13 MFCC coefficients from `signal` with default parameters.

 - `MFCC(signal, sampleRate, output, numCoeffs, numMelFilters, windowLength,
   windowStep, nFFT, lowFreq, highFreq, preEmphCoeff)`
   * Extract MFCC with different number coefficients dependings on the user specified
     parameters.

<!-- #rcurtin the table is a proposal I followed the same concept in
decision_tree.md, I think it is easier to use when we have several parameters,
please let me know what you think -->

### Functions Parameters:

<!-- @rcurtin, I was thinking of adding another column which is the value
ranges, I think most users will use the default values, but if someone would
like to change them they need to know that the highFreq cannot be 100 KHz
especially if they do not have any background in audio processing. -->

|     **name**     |  **type**   |   **default**  | **description**                                  |
|------------------|-------------|----------------|--------------------------------------------------|
| `signal`         | `arma::mat` | _(N/A)_ | Raw PCM audio samples.                                  |
| `sampleRate`     | `size_t`    | _(N/A)_ | Sample rate of the audio in Hz (e.g. `16000`, `44100`). |
| `output`         | `arma::mat` | _(N/A)_ | Output matrix of shape `(numMelFilters x numWindows)`.  |
| `numMelFilters`  | `size_t`    | `40`    | Number of mel-spaced triangular filters.                |
| `windowLength`   | `float`     | `25.0`  | Window length in milliseconds.                          |
| `windowStep`     | `float`     | `10.0`  | Window hop (step) in milliseconds.                      | 
| `nFFT`           | `size_t`    | `0`     | FFT size; `0` means the number of points fed to FFT is chosen automatically using the next power of 2 >= of the window length. |
| `lowFreq`        | `float`     | `0.0`   | Low frequency bound for the mel filterbank in Hz.       |
| `highFreq`       | `float`     | `0.0`   | High frequency bound in Hz; `0` means `sampleRate / 2`. |
| `preEmphCoeff`   | `float`     | `0.97`  | Finit impulse response filter coefficient; `0` disables the filter. |

***Note:*** Different types can be used for `signal` (e.g., `arma::fmat`, `arma::imat`).
However the signal needs to be represented in floating points. Therefore, if
MFE / MFCC filter is intended to be used, it is prefered to load the signal as
a floating point.

---

Apply MFE filter with default parameters on voice signals:

```c++
arma::mat signal;
mlpack::AudioOptions opts = mlpack::Fatal + mlpack::WAV;
mlpack::Load("voice.wav", signal, opts);

arma::mat mfe;
mlpack::MFE(signal, opts.SampleRate(), mfe);

std::cout << "MFE shape: " << mfe.n_rows << " x " << mfe.n_cols << std::endl;

```

Specifying a custom number of mel filters and frequency range:

```c++
arma::fmat signal;
mlpack::AudioOptions opts = mlpack::Fatal + mlpack::WAV;
mlpack::Load("voice.wav", signal, opts);

arma::fmat mfe;
// 80 mel filters, default window size, frequency range 300–8000 Hz.
mlpack::MFE(signal, opts.SampleRate(), mfe, 80, 25.0, 10.0, 0, 300.0,
    8000.0);
```
---

Extract 13 MFCCs from a WAV file.

```c++
arma::mat signal;
mlpack::AudioOptions opts = mlpack::WAV;
mlpack::Load("voice.wav", signal, opts);

arma::mat mfcc;
mlpack::MFCC(signal, opts.SampleRate(), mfcc);

// mfcc has a shape of 13 x numWindows.
std::cout << "MFCC shape: " << mfcc.n_rows << " x " << mfcc.n_cols
    << std::endl;
```

Extract 20 MFCCs with 80 mel filters from an MP3 file:

```c++
arma::fmat signal;
mlpack::AudioOptions opts = mlpack::MP3;
mlpack::Load("voice.mp3", signal, opts);

arma::fmat mfcc;
mlpack::MFCC(signal, opts.SampleRate(), mfcc, 20, 80);
```

MFCC with custom parameters:

```c++
arma::fmat signal;
mlpack::AudioOptions opts = mlpack::WAV;
mlpack::Load("voice.wav", signal, opts);

arma::fmat mfcc;
mlpack::MFCC(signal, opts.SampleRate(), mfcc, 13, 26, 25.0, 10.0, 512, 300.0, 3400.0, 0.97);
```
