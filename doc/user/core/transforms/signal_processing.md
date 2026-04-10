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
 * [Mel-Spectrogram and MFCCs | Lecture 72 (Part 1) | Applied Deep Learning](https://github.com/maziarraissi/Applied-Deep-Learning/blob/main/06%20-%20Speech%20%26%20Music/01%20-%20Recognition.pdf)

## `MFE()`

 - `MFE(signal, sampleRate, output)`
   * Extract log-mel filterbank energies from `signal` using default parameters.

 - `MFE(signal, sampleRate, output, numMelFilters, windowLength, windowStep,
   nFFT, lowFreq, highFreq)`
   * Extract log-mel filterbank energies depending on the user specified
     parameters.

### Functions Parameters:

|     **name**     |  **type**   |   **default**  | **description**                                         |
|------------------|-------------|----------------|---------------------------------------------------------|
| `signal`         | `arma::mat` | _(n/a)_ | raw pcm audio samples.                                         |
| `output`         | `arma::mat` | _(n/a)_ | output matrix of shape `(nummelfilters x numwindows)`.         |
| `sampleRate`     | `size_t`    | _(n/a)_ | sample rate of the audio in hz (e.g. `16000`, `44100`).        |
| `numMelFilters`  | `size_t`    | `40`    | number of mel-spaced triangular filters. Typical range (`20` to `100`) |
| `windowLength`   | `float`     | `25.0`  | window length in milliseconds.  Typical range (`20` to `40`)           |
| `windowStep`     | `float`     | `10.0`  | window hop (step) in milliseconds. Typical range (`5` to `20`)         |
| `nFFT`           | `size_t`    | `0`     | fft size; `0` means the number of points fed to fft is chosen automatically using the next power of 2 >= of the window length. Typical range (`256` to `4096`) |
| `lowFreq`        | `float`     | `0.0`   | low frequency bound for the mel filterbank in hz. Typical range (`0` to `300`) |
| `highFreq`       | `float`     | `0.0`   | high frequency bound in hz; `0` means `sampleRate / 2`. Typical range (`4000` to sampleRate / 2) |

***Note:*** if the audio signal was loaded as an integral type, you can either
change the loading code to load directly into a floating-point type matrix, or
use conv_to to convert to a floating-point type, then scale the range to [-1, 1]

***Note:*** If the audio signal has multiple channels, the channels must be
de-interleaved before calling `MFE()` with each column representing a separate
channel.  This preserves spatial information (e.g., which sounds come from the
left vs. right).  If spatial information is not needed, the channels can be
mixed down to mono before processing (e.g., `mono = (left + right) / 2`).
Both cases are demonstrated in the examples below.

---

Apply MFE filter with default parameters on voice signals:

```c++
// See https://datasets.mlpack.org/sine.wav
arma::mat signal;
mlpack::AudioOptions opts = mlpack::Fatal + mlpack::WAV;
mlpack::Load("sine.wav", signal, opts);

arma::mat mfe;
mlpack::MFE(signal, opts.SampleRate(), mfe);

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
mlpack::MFE(signal, opts.SampleRate(), mfe, 80, 25.0, 10.0, 0, 300.0,
    8000.0);
```
---

## `MFCC()`

 - `MFCC(signal, sampleRate, output)`
   * Extract 13 MFCC coefficients from `signal` with default parameters.

 - `MFCC(signal, output, sampleRate, numCoeffs, numMelFilters, windowLength,
   windowStep, nFFT, lowFreq, highFreq)`
   * Extract MFCC with different number coefficients dependings on the user specified
     parameters.

|     **name**     |  **type**   |   **default**  | **description**                                  |
|------------------|-------------|----------------|--------------------------------------------------|
| `signal`         | `arma::mat` | _(n/a)_ | raw pcm audio samples.                                  |
| `output`         | `arma::mat` | _(n/a)_ | output matrix of shape `(nummelfilters x numwindows)`.  |
| `sampleRate`     | `size_t`    | _(n/a)_ | sample rate of the audio in hz (e.g. `16000`, `44100`). |
| `numCoeff`       | `size_t`    | _(n/a)_ | Number of cepstral coefficients.                        |
| `numMelFilters`  | `size_t`    | `40`    | number of mel-spaced triangular filters.  Typical range (`20` to `100`)    |
| `windowLength`   | `float`     | `25.0`  | window length in milliseconds. Typical range (`20` to `40`)                |
| `windowStep`     | `float`     | `10.0`  | window hop (step) in milliseconds.  Typical range (`5` to `20`)            |
| `nFft`           | `size_t`    | `0`     | fft size; `0` means the number of points fed to fft is chosen automatically using the next power of 2 >= of the window length. Typical range (`256` to `4096`) |
| `lowFreq`        | `float`     | `0.0`   | low frequency bound for the mel filterbank in hz. Typical range (`0` to `300`)      |
| `highFreq`       | `float`     | `0.0`   | high frequency bound in hz; `0` means `sampleRate / 2`. Typical range (`4000` to sampleRate / 2) |

***Note:*** if the audio signal was loaded as an integral type, you can either
change the loading code to load directly into a floating-point type matrix, or
use conv_to to convert to a floating-point type, then scale the range to [-1, 1]

***Note:*** If the audio signal has multiple channels, the channels must be
de-interleaved before calling `MFCC()` with each column representing a separate
channel.  This preserves spatial information (e.g., which sounds come from the
left vs. right).  If spatial information is not needed, the channels can be
mixed down to mono before processing (e.g., `mono = (left + right) / 2`).
Both cases are demonstrated in the examples below.

**Note:*** `numCoeffs` must be less than or equal to `numMelFilters`.  The
DCT compresses `numMelFilters` log-mel energies down to `numCoeffs` cepstral
coefficients — it cannot produce more coefficients than there are input
values.

---

Extract 13 MFCCs from a WAV file.

```c++
// See https://datasets.mlpack.org/sine.wav
arma::mat signal;
mlpack::AudioOptions opts = mlpack::Fatal + mlpack::WAV;
mlpack::Load("sine.wav", signal, opts);

arma::mat mfcc;
mlpack::MFCC(signal, opts.SampleRate(), mfcc);

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
mlpack::MFCC(signal, opts.SampleRate(), mfcc, 20, 80);
```

MFCC with custom parameters:

```c++
// See https://datasets.mlpack.org/sine.wav
arma::fmat signal;
mlpack::AudioOptions opts = mlpack::Fatal + mlpack::WAV;
mlpack::Load("sine.wav", signal, opts);

arma::fmat mfcc;
mlpack::MFCC(signal, opts.SampleRate(), mfcc, 13, 26, 25.0, 10.0, 512, 300.0,
    3400.0);
```
