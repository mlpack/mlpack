/**
 * @file core/kernels/pspectrum_string_kernel.hpp
 * @author Ryan Curtin
 *
 * Implementation of the p-spectrum string kernel, created for use with FastMKS.
 * Instead of passing a data matrix to FastMKS which stores the kernels, pass a
 * one-dimensional data matrix (data vector) to FastMKS which stores indices of
 * strings; then, the actual strings are given to the PSpectrumStringKernel at
 * construction time, and the kernel knows to map the indices to actual strings.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_KERNELS_PSPECTRUM_STRING_KERNEL_HPP
#define MLPACK_CORE_KERNELS_PSPECTRUM_STRING_KERNEL_HPP

#include <map>
#include <string>
#include <vector>

#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/log.hpp>

namespace mlpack {

/**
 * The p-spectrum string kernel.  Given a length p, the p-spectrum kernel finds
 * the contiguous subsequence match count between two strings.  The kernel will
 * take every possible substring of length p of one string and count how many
 * times it appears in the other string.
 *
 * The string kernel, when created, must be passed a reference to a series of
 * string datasets (std::vector<std::vector<std::string> >&).  This is because
 * mlpack only supports datasets which are Armadillo matrices -- and a dataset
 * of variable-length strings cannot be easily cast into an Armadillo matrix.
 *
 * Therefore, once the PSpectrumStringKernel is created with a reference to the
 * string datasets, a "fake" Armadillo data matrix must be created, which simply
 * holds indices to the strings they represent.  This "fake" matrix has two rows
 * and n columns (where n is the number of strings in the dataset).  The first
 * row holds the index of the dataset (remember, the kernel can have multiple
 * datasets), and the second row holds the index of the string.  A fake matrix
 * containing only strings from dataset 0 might look like this:
 *
 * [[0 0 0 0 0 0 0 0 0]
 *  [0 1 2 3 4 5 6 7 8]]
 *
 * This fake matrix is then given to the machine learning method, which will
 * eventually call PSpectrumStringKernel::Evaluate(a, b), where a and b are two
 * columns of the fake matrix.  The string kernel will then map these fake
 * columns back to the strings they represent, and then correctly evaluate the
 * kernel.
 *
 * Unfortunately, not every machine learning method will work with this kernel.
 * Only machine learning methods which do not ever operate on the explicit
 * representation of points can use this kernel.  So, for instance, one cannot
 * build a kd-tree on strings, because the BinarySpaceTree<> class will split
 * the data according to the fake data matrix -- resulting in a meaningless
 * tree.  This kernel was originally written for the FastMKS method; so, at the
 * very least, it will work with that.
 */
class PSpectrumStringKernel
{
 public:
  /**
   * Initialize the PSpectrumStringKernel with the given string datasets.  For
   * more information on this, see the general class documentation.
   *
   * @param datasets Sets of string data.
   * @param p The length of substrings to search.
   */
  inline PSpectrumStringKernel(
      const std::vector<std::vector<std::string>>& datasets,
      const size_t p);

  /**
   * Evaluate the kernel for the string indices given.  As mentioned in the
   * class documentation, a and b should be 2-element vectors, where the first
   * element contains the index of the dataset and the second element contains
   * the index of the string.  Therefore, if [2 3] is passed for a, the string
   * used will be datasets[2][3] (datasets is of type
   * std::vector<std::vector<std::string> >&).
   *
   * @param a Index of string and dataset for first string.
   * @param b Index of string and dataset for second string.
   */
  template<typename VecType>
  double Evaluate(const VecType& a, const VecType& b) const;

  //! Access the lists of substrings.
  const std::vector<std::vector<std::map<std::string, int> > >& Counts() const
  { return counts; }
  //! Modify the lists of substrings.
  std::vector<std::vector<std::map<std::string, int> > >& Counts()
  { return counts; }

  //! Access the value of p.
  size_t P() const { return p; }
  //! Modify the value of p.
  size_t& P() { return p; }

 private:
  //! Mappings of the datasets to counts of substrings.  Such a huge structure
  //! is not wonderful...
  std::vector<std::vector<std::map<std::string, int> > > counts;

  //! The value of p to use in calculation.
  size_t p;
};

} // namespace mlpack

// Include implementation of templated Evaluate().
#include "pspectrum_string_kernel_impl.hpp"

#endif
