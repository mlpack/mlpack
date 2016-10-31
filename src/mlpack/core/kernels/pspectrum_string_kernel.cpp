/**
 * @file pspectrum_string_kernel.cpp
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
#include "pspectrum_string_kernel.hpp"

using namespace std;
using namespace mlpack;
using namespace mlpack::kernel;

/**
 * Initialize the PSpectrumStringKernel with the given string datasets.  For
 * more information on this, see the general class documentation.
 *
 * @param datasets Sets of string data.  @param p The length of substrings to
 * search.
 */
mlpack::kernel::PSpectrumStringKernel::PSpectrumStringKernel(
    const std::vector<std::vector<std::string> >& datasets,
    const size_t p) :
    datasets(datasets),
    p(p)
{
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
        string sub = str.substr(start, p);

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
