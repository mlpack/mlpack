#ifndef MLPACK_CORE_DATA_DATA_PRE_PROCESSING_IMPL_HPP
#define MLPACK_CORE_DATA_DATA_PRE_PROCESSING_IMPL_HPP

#include "data.hpp"

namespace mlpack {
namespace data {
  Data::Data(std::string file_path)
  {
    // Constructor
    path = file_path;
    mlpack::data::Load(path, data, true);
    data = data.submat(0, 1, data.n_rows - 1, data.n_cols - 1);
    getHeaders();
    getNumericAttributes();
  }

  void Data:: info()
  {
    // Returns Index range, list of headers
    std::cout << typeid(data).name() << std::endl;
    std::cout << "Index Range: " << data.n_cols << std::endl;
    std::cout << "Numeric Attributes: " << numericData.n_rows << std::endl;
    std::cout << "Non-numeric Attributes: " << headers.size() - numericData.n_rows << std::endl;
    std::cout << "Data Columns (total " << data.n_rows << "):" << std::endl;
    for (int i = 0; i < headers.size(); i++)
    {
      if(numericAttributes[i])
        std::cout << std::left << std::setw(20) << headers[i] << "numeric type" << std::endl;
      else  
        std::cout << std::left << std::setw(20) << headers[i] << "non-numeric type" << std::endl;
    }
    std::cout << std::endl;
  }

  void Data::getHeaders()
  {
    // Extract attribute headers
    std::fstream fin;
    fin.open(path, std::ios::in);
    std::string line, word, temp;
    getline(fin, line);
    std::stringstream s(line);
    while (getline(s, word, ','))
    {
      headers.push_back(word);
    }
  }

  bool Data::is_digits(const std::string& str)
  {
    // To check the datatype
    return str.find_first_not_of("-.0123456789") == std::string::npos;
  }

  void Data::getNumericAttributes()
  {
    // Classify attributes as numeric and non-numeric
    // For this we are checking only first row of values
    std::fstream fin;
    fin.open(path, std::ios::in);
    std::string line, word, temp;
    getline(fin, line);
    getline(fin, line);
    std::stringstream s(line);
    while (getline(s, word, ','))
    {
      if (is_digits(word))
        numericAttributes.push_back(true);
      else
        numericAttributes.push_back(false);
    }

    numericData = data;
    int removedCount = 0;
    for (int i = 0; i < numericAttributes.size(); i++)
    {
      if (!numericAttributes[i])
      {
        numericData.shed_row(i - removedCount);
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

  void Data::head()
  {
    // Top 5 rows
    infoHelper(4, 0);
  }

  void Data::tail()
  {
    // Last 5 rows
    infoHelper(data.n_cols - 1, data.n_cols - 5);
  }

  void Data::describe()
  {
    // Shows the attribute wise count, mean, std, min, max, variance
    arma::running_stat_vec<double> stats;

    for (int i = 0; i < numericData.n_cols; i++)
    {
      arma::mat row = numericData.submat(0, i, numericData.n_rows - 1, i);
      stats(arma::conv_to<arma::vec>::from(row));
    }
    
    std::cout.setf(std::ios::fixed);
    std::cout.width(5);
    std::cout << "Count: " << std::endl << stats.count() << std::endl;
    std::cout << "Mean: " << std::endl << stats.mean() << std::endl;
    std::cout << "Standard Deviation: " << std::endl << stats.stddev() << std::endl;
    std::cout << "Minimum: " << std::endl << stats.min() << std::endl;
    std::cout << "Maximum: " << std::endl << stats.max() << std::endl;
    std::cout << "Variance: " << std::endl << stats.var() << std::endl;
  }

} // namespace data
} // namespace mlpack

#endif
