/**
 * @file core/data/pre_processing_impl.hpp
 * @author Gopi Manohar Tatiraju
 *
 * Generic dataframe-like class for C++ for data pre processing
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_DATA_PRE_PROCESSING_IMPL_HPP
#define MLPACK_CORE_DATA_DATA_PRE_PROCESSING_IMPL_HPP

#include "pre_processing.hpp"

namespace mlpack {
namespace data {

/**
 * A datafram-like class for pre-processing the dataset
 * before feeding it to any model.
 *
 * @param path Path to the csv file
 * @param headers Headers of the dataset
 * @param numericAttributes Bool array to indicate numeric attributes
 * @param numericData Arma mat of numeric attributes only
 */

Data::Data(std::string file_path)
  {
    path = file_path;
    data.load(arma::csv_name(path, headers, arma::csv_opts::trans));
    mlpack::data::Load(path, data, true);
    data = data.submat(0, 1, data.n_rows - 1, data.n_cols - 1);
    numericData = data;
    getNumericAttributes();
  }

void Data:: Info()
  {
    // Returns Index range, list of headers
    std::cout << typeid(data).name() << std::endl;
    std::cout << "Index Range: " << data.n_cols << std::endl;
    std::cout << "Numeric Attributes: " << numericData.n_rows << std::endl;
    std::cout << "Non-numeric Attributes: " << headers.n_rows - numericData.n_rows << std::endl;
    std::cout << "Data Dimensions (total " << data.n_rows << "):" << std::endl;
    for (int i = 0; i < headers.size(); i++)
    {
      if(numericAttributes[i])
        std::cout << std::left << std::setw(20) << headers[i] << "numeric type" << std::endl;
      else  
        std::cout << std::left << std::setw(20) << headers[i] << "non-numeric type" << std::endl;
    }
    std::cout << std::endl;
  }

bool Data::isDigits(const std::string& str)
  {
    // To check the datatype
    return str.find_first_not_of("-.0123456789") == std::string::npos;
  }

void Data::getNumericAttributes()
{
  std::fstream fin;
  fin.open(path, std::ios::in);
  std::string line, word, temp;
  getline(fin, line);
  getline(fin, line);
  std::stringstream s(line);
  while (getline(s, word, ','))
  {
    if (isDigits(word))
      numericAttributes.push_back(true);
    else
      numericAttributes.push_back(false);
  }

  int removedCount = 0;
  for (int i = 0; i < numericAttributes.size(); i++)
  {
    if (!numericAttributes[i])
    {
      int f = i - removedCount;
      numericData.shed_row(f);
      removedCount++;
    }
  }
}

void Data::infoHelper(int lower, int upper)
  {
    for (auto i : headers)
    {
      std::cout << i << " ";
    }
    std::cout << std::endl << std::endl;
    std::cout.precision(2);
    std::cout.setf(std::ios::fixed);
    std::cout.width(15);
    data.submat(0, upper, data.n_rows - 1, lower).t().raw_print(std::cout);
  }

void Data::Head()
  {
    // Top 5 rows
    infoHelper(4, 0);
  }

void Data::Tail()
  {
    // Last 5 rows
    infoHelper(data.n_cols - 1, data.n_cols - 5);
  }

void Data::Describe()
  {
    // Shows the attribute wise count, mean, std, min, max, variance
    // Find a way to add the attributes as well
    /** Current output looks something like this(formatting and spacing can be done later)
     *   Count: 9
     *   Mean    Std     Minimum Maximum Variance
     *   3.00      1.22      2.00      6.00      1.50
     *   900000.00 531830.57      0.001515000.00282843750000.00
     *   3047.78     15.89   3021.00   3067.00    252.44
     *   3261.67    914.82   1543.00   4019.00 836886.50
     *   8.01      4.10      3.00     14.00     16.79
     */

     /** Will modify it to(still figuring out a way for this)
      *   Count: 9
      *   Attribute    Mean    Std     Minimum Maximum Variance
      *   attribute1   3.00      1.22      2.00      
      *   attribute1   900000.00 531830.57      0.00151500
      *   attribute1   3047.78     15.89   3021.00   3
      *   attribute1   3261.67    914.82   1543.00   
      *   attribute1   8.01      4.10      3.00     
      */

    arma::running_stat_vec<double> stats;

    for (int i = 0; i < numericData.n_cols; i++)
    {
      arma::mat row = numericData.submat(0, i, numericData.n_rows - 1, i);
      stats(arma::conv_to<arma::vec>::from(row));
    }
    
    std::cout << "Count: " << stats.count() << std::endl;

    arma::mat ab = arma::join_rows(stats.mean(), stats.stddev(), 
                                    stats.min(), stats.max());
    ab = arma::join_rows(ab, stats.var());
    std::cout << "\t\tMean" << "\t\tStd" << "\t\tMinimum" << "\t\tMaximum" 
              << "\t\tVariance" << std::endl;
    std::cout.setf(std::ios::fixed);
    std::cout.precision(2);
    std::cout.width(20);
    ab.raw_print(std::cout);
    //for (int i = 0; i < headers.size(); i++)
    //{
    //  if (numericAttributes[i])
    //    std::cout << "\t" << headers[i];
    //}

    //std::cout << std::endl;
  }

void Data::valueCounts(int colNumber)
  {
    std::fstream fin;
    fin.open(path, std::ios::in);
    std::string line;
    std::map<std::string, int> roo;

    // To skip the header
    getline(fin, line);

    while (fin)
    {
      getline(fin, line);
      int count = 0;
      std::string colValue = "";

      for (int i = 0; i < line.size(); i++)
      {
        if (line[i] == ',')
        {
          count++;
          continue;
        }
        if (count > colNumber)
        {
          if (roo.find(colValue) == roo.end())
            roo.insert({ colValue, 0 });
          if (roo.find(colValue) != roo.end())
            roo[colValue]++;
          break;
        }
        if (count == colNumber)
          colValue = colValue + line[i];
      }
    }

    for (auto i : roo)
      std::cout << i.first << " " << i.second << std::endl;

    fin.close();
  }

void Data::getNumericData()
{
  for (int i = 0; i < headers.size(); i++)
  {
    if (numericAttributes[i])
      std::cout << headers[i] << " ";
  }
  std::cout << std::endl;
  std::cout.precision(2);
  std::cout.width(15);
  std::cout.setf(std::ios::fixed);
  numericData.t().raw_print(std::cout);
}
} // namespace data
} // namespace mlpack

#endif
