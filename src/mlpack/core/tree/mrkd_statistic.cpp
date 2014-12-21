/**
 * @file mrkd_statistic.cpp
 * @author James Cline
 *
 * Definition of the statistic for multi-resolution kd-trees.
 *
 * This file is part of MLPACK 1.0.9.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
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
