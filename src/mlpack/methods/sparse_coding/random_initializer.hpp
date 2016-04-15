/**
 * @file random_initializer.hpp
 * @author Nishant Mehta
 *
 * A very simple random dictionary initializer for SparseCoding; it is probably
 * not a very good choice.
 */
#ifndef MLPACK_METHODS_SPARSE_CODING_RANDOM_INITIALIZER_HPP
#define MLPACK_METHODS_SPARSE_CODING_RANDOM_INITIALIZER_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace sparse_coding {

/**
 * A DictionaryInitializer for use with the SparseCoding class.  This provides a
 * random, normally distributed dictionary, such that each atom has a norm of 1.
 */
class MLPACK_API RandomInitializer
{
 public:
  /**
   * Initialize the dictionary randomly from a normal distribution, such that
   * each atom has a norm of 1.  This is simple enough to be included with the
   * definition.
   *
   * @param data Dataset to use for initialization.
   * @param atoms Number of atoms (columns) in the dictionary.
   * @param dictionary Dictionary to initialize.
   */
  static void Initialize(const arma::mat& data,
                         const size_t atoms,
                         arma::mat& dictionary)
  {
    // Create random dictionary.
    dictionary.randn(data.n_rows, atoms);

    // Normalize each atom.
    for (size_t j = 0; j < atoms; ++j)
      dictionary.col(j) /= norm(dictionary.col(j), 2);
  }
};

} // namespace sparse_coding
} // namespace mlpack

#endif
