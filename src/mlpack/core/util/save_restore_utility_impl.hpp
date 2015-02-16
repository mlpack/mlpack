/**
 * @file save_restore_utility_impl.hpp
 * @author Neil Slagle
 *
 * The SaveRestoreUtility provides helper functions in saving and
 *   restoring models.  The current output file type is XML.
 */
#ifndef __MLPACK_CORE_UTIL_SAVE_RESTORE_UTILITY_IMPL_HPP
#define __MLPACK_CORE_UTIL_SAVE_RESTORE_UTILITY_IMPL_HPP

// In case it hasn't been included already.
#include "save_restore_utility.hpp"
#include "log.hpp"

namespace mlpack {
namespace util {

template<typename T>
std::vector<T>& SaveRestoreUtility::LoadParameter(std::vector<T>& v,
                                                  const std::string& name) const
{
  std::map<std::string, std::string>::const_iterator it = parameters.find(name);
  if (it != parameters.end())
  {
    v.clear();
    std::string value = (*it).second;
    boost::char_separator<char> sep (",");
    boost::tokenizer<boost::char_separator<char> > tok (value, sep);
    std::list<std::list<double> > rows;
    for (boost::tokenizer<boost::char_separator<char> >::iterator
        tokIt = tok.begin(); tokIt != tok.end(); ++tokIt)
    {
      T t;
      std::istringstream iss(*tokIt);
      iss >> t;
      v.push_back(t);
    }
  }
  else
  {
    Log::Fatal << "LoadParameter(): node '" << name << "' not found.\n";
  }
  return v;
}

// Load Armadillo matrices specially, in order to preserve precision.  This
// catches dense objects.
template<typename eT>
arma::Mat<eT>& SaveRestoreUtility::LoadParameter(
    arma::Mat<eT>& t,
    const std::string& name) const
{
  std::map<std::string, std::string>::const_iterator it = parameters.find(name);
  if (it != parameters.end())
  {
    std::string value = (*it).second;
    std::istringstream input(value);

    std::string err; // Store a possible error message.
    if (!arma::diskio::load_csv_ascii(t, input, err))
    {
      Log::Fatal << "LoadParameter(): error while loading node '" << name
          << "': " << err << ".\n";
    }
  }
  else
  {
    Log::Fatal << "LoadParameter(): node '" << name << "' not found.\n";
  }
  return t;
}

// Load Armadillo matrices specially, in order to preserve precision.  This
// catches sparse objects.
template<typename eT>
arma::SpMat<eT>& SaveRestoreUtility::LoadParameter(
    arma::SpMat<eT>& t,
    const std::string& name) const
{
  std::map<std::string, std::string>::const_iterator it = parameters.find(name);
  if (it != parameters.end())
  {
    std::string value = (*it).second;
    std::istringstream input(value);

    std::string err; // Store a possible error message.
    if (!arma::diskio::load_coord_ascii(t, input, err))
    {
      Log::Fatal << "LoadParameter(): error while loading node '" << name
          << "': " << err << ".\n";
    }
  }
  else
  {
    Log::Fatal << "LoadParameter(): node '" << name << "' not found.\n";
  }
  return t;
}

template<typename T>
T& SaveRestoreUtility::LoadParameter(
    T& t,
    const std::string& name,
    const typename boost::enable_if_c<(!arma::is_arma_type<T>::value &&
                                       !arma::is_arma_sparse_type<T>::value)
                                     >::type* /* junk */) const
{
  std::map<std::string, std::string>::const_iterator it = parameters.find(name);
  if (it != parameters.end())
  {
    std::string value = (*it).second;
    std::istringstream input(value);
    input >> t;
    return t;
  }
  else
  {
    Log::Fatal << "LoadParameter(): node '" << name << "' not found.\n";
  }
  return t;
}

// Print Armadillo matrices specially, in order to preserve precision.  This
// catches dense objects.
template<typename eT, typename T1>
void SaveRestoreUtility::SaveParameter(
    const arma::Base<eT, T1>& t,
    const std::string& name)
{
  // Create a matrix to give to save_csv_ascii().  This may incur a copy,
  // depending on the compiler's intelligence.  But the disk bandwidth is going
  // to be the main slowdown anyway...
  arma::Mat<eT> temp(t.get_ref());

  // Use save_csv_ascii().  This is *slightly* imprecise and it may be better to
  // store this raw.  But this is readable...
  std::ostringstream output;
  arma::diskio::save_csv_ascii(temp, output);
  parameters[name] = output.str();
}

// Print sparse Armadillo matrices specially, in order to preserve precision.
// This catches sparse objects.
template<typename eT, typename T1>
void SaveRestoreUtility::SaveParameter(
    const arma::SpBase<eT, T1>& t,
    const std::string& name)
{
  // Create a matrix to give to save_coord_ascii().  This may incur a copy,
  // depending on the compiler's intelligence.  But the disk bandwidth is going
  // to be the main slowdown anyway...
  arma::SpMat<eT> temp(t.get_ref());

  // Use save_coord_ascii().  This is *slightly* imprecise and it may be better
  // to store this raw.  But this is readable...
  std::ostringstream output;
  arma::diskio::save_coord_ascii(temp, output);
  parameters[name] = output.str();
}

template<typename T>
void SaveRestoreUtility::SaveParameter(
    const T& t,
    const std::string& name,
    const typename boost::enable_if_c<(!arma::is_arma_type<T>::value &&
                                       !arma::is_arma_sparse_type<T>::value)
                                     >::type* /* junk */)
{
  std::ostringstream output;
  // Manually increase precision to solve #313 for now, until we have a way to
  // store this as an actual binary number.
  output << std::setprecision(15) << t;
  parameters[name] = output.str();
}

template<typename T>
void SaveRestoreUtility::SaveParameter(const std::vector<T>& t,
                                       const std::string& name)
{
  std::ostringstream output;
  for (size_t index = 0; index < t.size(); ++index)
  {
    output << t[index] << ",";
  }
  std::string vectorAsStr = output.str();
  vectorAsStr.erase(vectorAsStr.length() - 1);
  parameters[name] = vectorAsStr;
}

}; // namespace util
}; // namespace mlpack

#endif
