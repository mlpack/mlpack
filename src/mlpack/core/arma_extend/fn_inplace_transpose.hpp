
/**
 * This method automatically selects the transpose method,
 * depending on available memory.
 */
template<typename eT>
bool
inline
inplace_transpose(Mat<eT>& X)
  {
    try
    {
      X = trans(X);
      return false;
    }
    catch (std::bad_alloc& exception)
    {
      inplace_trans(X, "lowmem");
      return true;
    }
  }
