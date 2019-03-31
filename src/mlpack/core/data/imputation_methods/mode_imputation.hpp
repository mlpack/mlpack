/**
 * @file mode_imputation.hpp
 * @author Gaurav Sharma
 *
 * Definition and Implementation of the ModeImputation class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_IMPUTE_STRATEGIES_MODE_IMPUTATION_HPP
#define MLPACK_CORE_DATA_IMPUTE_STRATEGIES_MODE_IMPUTATION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace data {
/**
 * A simple mode imputation class
 * @tparam T Type of armadillo matrix
 */
template <typename T>
class ModeImputation
{
 public:
  /**
   * Impute function searches through the input looking for mappedValue and
   * replaces it with the mode of the given dimension if exists otherwise it
   * will throw exception. The result is overwritten to the input matrix.
   *
   * @param input Matrix that contains mappedValue.
   * @param mappedValue Value that the user wants to get rid of.
   * @param dimension Index of the dimension of the mappedValue.
   * @param columnMajor State of whether the input matrix is columnMajor or not.
   */
  void Impute(arma::Mat<T>& input,
              const T& mappedValue,
              const size_t dimension,
              const bool columnMajor = true)
  {
    double sum = 0;
    size_t elems = 0; // excluding nan or missing target

    using PairType = std::pair<size_t, size_t>;
    // dimensions and indexes are saved as pairs inside this vector.
    std::vector<PairType> targets;
    // good elements are kept inside this vector.
    std::vector<double> elemsToKeep;

    if (columnMajor)
    {
      for (size_t i = 0; i < input.n_cols; ++i)
      {
        if (input(dimension, i) == mappedValue ||
            std::isnan(input(dimension, i)))
        {
          targets.emplace_back(dimension, i);
        }
        else
        {
          elemsToKeep.push_back(input(dimension, i));
        }
      }
    }
    else
    {
      for (size_t i = 0; i < input.n_rows; ++i)
      {
        if (input(i, dimension) == mappedValue ||
            std::isnan(input(i, dimension)))
        {
          targets.emplace_back(i, dimension);
        }
        else
        {
         elemsToKeep.push_back(input(i, dimension));
        }
      }
    }

    // calculate mode
    // elemsWithFreq contains the elements of elemsToKeep
    // with their respective frequency.
    std::vector<PairType> elemsWithFreq;

    // modeCountPair is a PairType that contains the mode
    // and it's count(number of occurences).

    for (size_t i = 0; i < elemsToKeep.size(); ++i)
    {
     bool flag = true;

     for (PairType &elems : elemsWithFreq)
     {
      if (elemsToKeep[i] == elems.first)
      {
       elems.second += 1;     // increase the frequency by 1
       flag = false;
       break;
      }
     }

     if (flag)
     {
      // if the current element of elemsToKeep
      // is not present in elemsWithFreq then add it.
      elemsWithFreq.emplace_back(elemsToKeep[i], 1);
     }
    }

    PairType modeCountPair = elemsWithFreq[0];

    bool flag = false;

    for (size_t i = 1; i < elemsWithFreq.size(); i++)
    {
     if (elemsWithFreq[i].second > modeCountPair.second)
     {
      modeCountPair = elemsWithFreq[i];

      if (flag)
       flag = false;
     }
     else if (elemsWithFreq[i].second == modeCountPair.second)
     {
      flag = true;
     }
    }

    if (flag)
    {
     throw std::invalid_argument("Mode is not present");
    }

    // Now replace the calculated mode to the missing variables.
    // It only needs to loop through targets vector, not the whole matrix.
    for (const PairType &target : targets)
    {
     input(target.first, target.second) = modeCountPair.first;
    }
  }
}; // class ModeImputation

} // namespace data
} // namespace mlpack

#endif
