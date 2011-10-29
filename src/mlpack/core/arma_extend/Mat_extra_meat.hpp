/**
 * @file Mat_extra_meat.hpp
 * @author Ryan Curtin
 *
 * Extra overload of load() to allow transposition of matrix at load time.
 */

template<typename eT>
inline
bool
Mat<eT>::load(const std::string name, const file_type type, const bool print_status, const bool transpose)
  {
  bool result = load(name, type, print_status);

  if (transpose)
    {
    // Now transpose the matrix.
    *this = trans(*this);
    }

  return result;
  }

template<typename eT>
inline
bool
Mat<eT>::load(std::istream& is, const file_type type, const bool print_status, const bool transpose)
  {
  bool result = load(is, type, print_status);

  if (transpose)
    {
    // Now transpose the matrix.
    *this = trans(*this);
    }

  return result;
  }

template<typename eT>
inline
bool
Mat<eT>::save(const std::string name, const file_type type, const bool print_status, const bool transpose)
  {
  if (transpose)
    {
    // Save a temporary matrix.
    Mat<eT> tmp = trans(*this);

    return tmp.save(name, type, print_status);
    }
  else
    {
    return save(name, type, print_status);
    }
  }

template<typename eT>
inline
bool
Mat<eT>::save(std::ostream& os, const file_type type, const bool print_status, const bool transpose)
  {
  if (transpose)
    {
    // Save a temporary matrix.
    Mat<eT> tmp = trans(*this);

    return tmp.save(os, type, print_status);
    }
  else
    {
    return save(os, type, print_status);
    }
  }
