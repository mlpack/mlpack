/**
 * @file mrkd_statistic.cpp
 * @author James Cline
 *
 * Definition of the statistic for multi-resolution kd-trees.
 *
 * This file is part of mlpack 1.0.12.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "mrkd_statistic.hpp"

using namespace mlpack;
using namespace mlpack::tree;

MRKDStatistic::MRKDStatistic() :
    dataset(NULL),
    begin(0),
    count(0),
    leftStat(NULL),
    rightStat(NULL),
    parentStat(NULL)
{ }

/**
 * Returns a string representation of this object.
 */
std::string MRKDStatistic::ToString() const
{
  std::ostringstream convert;

  convert << "MRKDStatistic [" << this << std::endl;
  convert << "begin: " << begin << std::endl;
  convert << "count: " << count << std::endl;
  convert << "sumOfSquaredNorms: " << sumOfSquaredNorms << std::endl;
  if (leftStat != NULL)
  {
    convert << "leftStat:" << std::endl;
    convert << mlpack::util::Indent(leftStat->ToString());
  }
  if (rightStat != NULL)
  {
    convert << "rightStat:" << std::endl;
    convert << mlpack::util::Indent(rightStat->ToString());
  }
  return convert.str();
}
