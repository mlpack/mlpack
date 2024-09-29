/**
 * @file core/kernels/pspectrum_string_kernel_impl.hpp
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
#ifndef MLPACK_CORE_KERNELS_PSPECTRUM_STRING_KERNEL_IMPL_HPP
#define MLPACK_CORE_KERNELS_PSPECTRUM_STRING_KERNEL_IMPL_HPP

// In case it has not been included yet.
#include "pspectrum_string_kernel.hpp"

namespace mlpack {

inline PSpectrumStringKernel::PSpectrumStringKernel(
    const std::vector<std::vector<std::string> >& datasets,
    const size_t p) : p(p)
{
  if (p == 0)
  {
    throw std::invalid_argument(
        "PSpectrumStringKernel::PSpectrumStringKernel(): p must be positive");
  }

  // We have to assemble the counts of substrings.  This is not a particularly
  // fast operation, unfortunately, but it only needs to be done once.
  Log::Info << "Assembling counts of substrings of length " << p << "."
      << std::endl;

  // Resize for number of datasets.
  counts.resize(datasets.size());

  for (size_t dataset = 0; dataset < datasets.size(); ++dataset)
  {
    const std::vector<std::string>& set = datasets[dataset];

    // Resize for number of strings in dataset.
    counts[dataset].resize(set.size());

    // Inspect each string in the dataset.
    for (size_t index = 0; index < set.size(); ++index)
    {
      // Convenience references.
      const std::string& str = set[index];
      std::map<std::string, int>& mapping = counts[dataset][index];

      size_t start = 0;
      while ((start + p) <= str.length())
      {
        std::string sub = str.substr(start, p);

        // Convert all characters to lowercase.
        bool invalid = false;
        for (size_t j = 0; j < p; ++j)
        {
          if (!isalnum(sub[j]))
          {
            invalid = true;
            break; // Only consider substrings with alphanumerics.
          }

          sub[j] = tolower(sub[j]);
        }

        // Increment position in string.
        ++start;

        if (!invalid)
        {
          // Add to the map.
          ++mapping[sub];
        }
      }
    }
  }
  Log::Info << "Substring extraction complete." << std::endl;
}

/**
 * Evaluate the kernel for the string indices given.  As mentioned in the class
 * documentation, a and b should be 2-element vectors, where the first element
 * contains the index of the dataset and the second element contains the index
 * of the string.  Therefore, if [2 3] is passed for a, the string used will be
 * datasets[2][3] (datasets is of type std::vector<std::vector<std::string> >&).
 *
 * @param a Index of string and dataset for first string.
 * @param b Index of string and dataset for second string.
 */
template<typename VecType>
double PSpectrumStringKernel::Evaluate(const VecType& a,
                                       const VecType& b) const
{
  // Get the map of substrings for the two strings we are interested in.
  const std::map<std::string, int>& aMap = counts[a[0]][a[1]];
  const std::map<std::string, int>& bMap = counts[b[0]][b[1]];

  double eval = 0;

  // Loop through the two maps (which, when iterated through, are sorted
  // alphabetically).
  std::map<std::string, int>::const_iterator aIt = aMap.begin();
  std::map<std::string, int>::const_iterator bIt = bMap.begin();

  while ((aIt != aMap.end()) && (bIt != bMap.end()))
  {
    // Compare alphabetically (this is how std::map is ordered).
    int result = (*aIt).first.compare((*bIt).first);

    if (result == 0) // The same substring.
    {
      eval += ((*aIt).second * (*bIt).second);

      // Now increment both.
      ++aIt;
      ++bIt;
    }
    else if (result > 0)
    {
      // aIt is "ahead" of bIt (alphabetically); so increment bIt to "catch up".
      ++bIt;
    }
    else
    {
      // bIt is "ahead" of aIt (alphabetically); so increment aIt to "catch up".
      ++aIt;
    }
  }

  return eval;
}
} // namespace mlpack

#endif
