#include <chrono>
#include <mlpack.hpp>

using namespace mlpack;

void floating()
{
  size_t sampleRate = 16000;
  size_t oneHour = sampleRate * 3600;
  float pi = 3.141592f;

  arma::fvec t = arma::linspace<arma::fvec>(0, 1.0, oneHour);
  arma::fmat input = arma::sin(2.0 * pi * 440.0 * t);

  std::cout << "Input: shape (" << input.n_rows << " x " << input.n_cols
      << ")  range [" << input.min() << ", " << input.max() << "]"
      << std::endl;

  input.brief_print("sine wave signal:");

  arma::fmat mfe;
  double bestTime = 1e9;
  for (int trial = 0; trial < 3; ++trial)
  {
    auto start = std::chrono::high_resolution_clock::now();
    MFE(input, mfe, sampleRate);
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    bestTime = std::min(bestTime, ms);
  }
  std::cout << "Best float time is : " << bestTime << " ms" << std::endl;
}

void doubling()
{
  size_t sampleRate = 16000;
  size_t oneHour = sampleRate * 3600;
  double pi = 3.141592;

  arma::vec t = arma::linspace<arma::vec>(0, 1.0, oneHour);
  arma::mat input = arma::sin(2.0 * pi * 440.0 * t);

  std::cout << "Input: shape (" << input.n_rows << " x " << input.n_cols
      << ")  range [" << input.min() << ", " << input.max() << "]"
      << std::endl;

  input.brief_print("sine wave signal:");

  arma::mat mfe;
  double bestTime = 1e9;
  for (int trial = 0; trial < 3; ++trial)
  {
    auto start = std::chrono::high_resolution_clock::now();
    MFE(input, mfe, sampleRate);
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    bestTime = std::min(bestTime, ms);
  }
  std::cout << "Best double time is : " << bestTime << " ms" << std::endl;
} 

int main()
{
//  floating();

  doubling();

  return 0;
}
