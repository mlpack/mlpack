
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_julia_julia_util.cpp:

Program Listing for File julia_util.cpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_julia_julia_util.cpp>` (``/home/aakash/mlpack/src/mlpack/bindings/julia/julia_util.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <mlpack/bindings/julia/julia_util.h>
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/util/io.hpp>
   #include <stdint.h>
   
   using namespace mlpack;
   
   extern "C" {
   
   void IO_RestoreSettings(const char* programName)
   {
     IO::RestoreSettings(programName);
   }
   
   void IO_SetParamInt(const char* paramName, int paramValue)
   {
     IO::GetParam<int>(paramName) = paramValue;
     IO::SetPassed(paramName);
   }
   
   void IO_SetParamDouble(const char* paramName, double paramValue)
   {
     IO::GetParam<double>(paramName) = paramValue;
     IO::SetPassed(paramName);
   }
   
   void IO_SetParamString(const char* paramName, const char* paramValue)
   {
     IO::GetParam<std::string>(paramName) = paramValue;
     IO::SetPassed(paramName);
   }
   
   void IO_SetParamBool(const char* paramName, bool paramValue)
   {
     IO::GetParam<bool>(paramName) = paramValue;
     IO::SetPassed(paramName);
   }
   
   void IO_SetParamVectorStrLen(const char* paramName,
                                const size_t length)
   {
     IO::GetParam<std::vector<std::string>>(paramName).clear();
     IO::GetParam<std::vector<std::string>>(paramName).resize(length);
     IO::SetPassed(paramName);
   }
   
   void IO_SetParamVectorStrStr(const char* paramName,
                                const char* str,
                                const size_t element)
   {
     IO::GetParam<std::vector<std::string>>(paramName)[element] =
         std::string(str);
   }
   
   void IO_SetParamVectorInt(const char* paramName,
                             int* ints,
                             const size_t length)
   {
     // Create a std::vector<int> object; unfortunately this requires copying the
     // vector elements.
     std::vector<int> vec;
     vec.resize(length);
     for (size_t i = 0; i < length; ++i)
       vec[i] = ints[i];
   
     IO::GetParam<std::vector<int>>(paramName) = std::move(vec);
     IO::SetPassed(paramName);
   }
   
   void IO_SetParamMat(const char* paramName,
                       double* memptr,
                       const size_t rows,
                       const size_t cols,
                       const bool pointsAsRows)
   {
     // Create the matrix as an alias.
     arma::mat m(memptr, arma::uword(rows), arma::uword(cols), false, true);
     IO::GetParam<arma::mat>(paramName) = pointsAsRows ? m.t() : std::move(m);
     IO::SetPassed(paramName);
   }
   
   void IO_SetParamUMat(const char* paramName,
                        size_t* memptr,
                        const size_t rows,
                        const size_t cols,
                        const bool pointsAsRows)
   {
     // Create the matrix as an alias.
     arma::Mat<size_t> m(memptr, arma::uword(rows), arma::uword(cols), false,
         true);
     IO::GetParam<arma::Mat<size_t>>(paramName) = pointsAsRows ? m.t() :
         std::move(m);
     IO::SetPassed(paramName);
   }
   
   void IO_SetParamRow(const char* paramName,
                       double* memptr,
                       const size_t cols)
   {
     arma::rowvec m(memptr, arma::uword(cols), false, true);
     IO::GetParam<arma::rowvec>(paramName) = std::move(m);
     IO::SetPassed(paramName);
   }
   
   void IO_SetParamURow(const char* paramName,
                        size_t* memptr,
                        const size_t cols)
   {
     arma::Row<size_t> m(memptr, arma::uword(cols), false, true);
     IO::GetParam<arma::Row<size_t>>(paramName) = std::move(m);
     IO::SetPassed(paramName);
   }
   
   void IO_SetParamCol(const char* paramName,
                       double* memptr,
                       const size_t rows)
   {
     arma::vec m(memptr, arma::uword(rows), false, true);
     IO::GetParam<arma::vec>(paramName) = std::move(m);
     IO::SetPassed(paramName);
   }
   
   void IO_SetParamUCol(const char* paramName,
                        size_t* memptr,
                        const size_t rows)
   {
     arma::Col<size_t> m(memptr, arma::uword(rows), false, true);
     IO::GetParam<arma::Col<size_t>>(paramName) = std::move(m);
     IO::SetPassed(paramName);
   }
   
   void IO_SetParamMatWithInfo(const char* paramName,
                               bool* dimensions,
                               double* memptr,
                               const size_t rows,
                               const size_t cols,
                               const bool pointsAreRows)
   {
     data::DatasetInfo d(pointsAreRows ? cols : rows);
     for (size_t i = 0; i < d.Dimensionality(); ++i)
     {
       d.Type(i) = (dimensions[i]) ? data::Datatype::categorical :
           data::Datatype::numeric;
     }
   
     arma::mat m(memptr, arma::uword(rows), arma::uword(cols), false, true);
     std::get<0>(IO::GetParam<std::tuple<data::DatasetInfo, arma::mat>>(
         paramName)) = std::move(d);
     std::get<1>(IO::GetParam<std::tuple<data::DatasetInfo, arma::mat>>(
         paramName)) = pointsAreRows ? m.t() : std::move(m);
     IO::SetPassed(paramName);
   }
   
   int IO_GetParamInt(const char* paramName)
   {
     return IO::GetParam<int>(paramName);
   }
   
   double IO_GetParamDouble(const char* paramName)
   {
     return IO::GetParam<double>(paramName);
   }
   
   const char* IO_GetParamString(const char* paramName)
   {
     return IO::GetParam<std::string>(paramName).c_str();
   }
   
   bool IO_GetParamBool(const char* paramName)
   {
     return IO::GetParam<bool>(paramName);
   }
   
   size_t IO_GetParamVectorStrLen(const char* paramName)
   {
     return IO::GetParam<std::vector<std::string>>(paramName).size();
   }
   
   const char* IO_GetParamVectorStrStr(const char* paramName, const size_t i)
   {
     return IO::GetParam<std::vector<std::string>>(paramName)[i].c_str();
   }
   
   size_t IO_GetParamVectorIntLen(const char* paramName)
   {
     return IO::GetParam<std::vector<int>>(paramName).size();
   }
   
   int* IO_GetParamVectorIntPtr(const char* paramName)
   {
     const size_t size = IO::GetParam<std::vector<int>>(paramName).size();
     int* ints = new int[size];
   
     for (size_t i = 0; i < size; ++i)
       ints[i] = IO::GetParam<std::vector<int>>(paramName)[i];
   
     return ints;
   }
   
   size_t IO_GetParamMatRows(const char* paramName)
   {
     return IO::GetParam<arma::mat>(paramName).n_rows;
   }
   
   size_t IO_GetParamMatCols(const char* paramName)
   {
     return IO::GetParam<arma::mat>(paramName).n_cols;
   }
   
   double* IO_GetParamMat(const char* paramName)
   {
     // Are we using preallocated memory?  If so we have to handle this more
     // carefully.
     arma::mat& mat = IO::GetParam<arma::mat>(paramName);
     if (mat.n_elem <= arma::arma_config::mat_prealloc)
     {
       // Copy the memory to something that we can give back to Julia.
       double* newMem = new double[mat.n_elem];
       arma::arrayops::copy(newMem, mat.mem, mat.n_elem);
       return newMem; // We believe Julia will free it.  Hopefully we are right.
     }
     else
     {
       arma::access::rw(mat.mem_state) = 1;
       #if ARMA_VERSION_MAJOR >= 10
         arma::access::rw(mat.n_alloc) = 0;
       #endif
       return mat.memptr();
     }
   }
   
   size_t IO_GetParamUMatRows(const char* paramName)
   {
     return IO::GetParam<arma::Mat<size_t>>(paramName).n_rows;
   }
   
   size_t IO_GetParamUMatCols(const char* paramName)
   {
     return IO::GetParam<arma::Mat<size_t>>(paramName).n_cols;
   }
   
   size_t* IO_GetParamUMat(const char* paramName)
   {
     arma::Mat<size_t>& mat = IO::GetParam<arma::Mat<size_t>>(paramName);
   
     // Are we using preallocated memory?  If so we have to handle this more
     // carefully.
     if (mat.n_elem <= arma::arma_config::mat_prealloc)
     {
       // Copy the memory to something that we can give back to Julia.
       size_t* newMem = new size_t[mat.n_elem];
       arma::arrayops::copy(newMem, mat.mem, mat.n_elem);
       return newMem; // We believe Julia will free it.  Hopefully we are right.
     }
     else
     {
       arma::access::rw(mat.mem_state) = 1;
       #if ARMA_VERSION_MAJOR >= 10
         arma::access::rw(mat.n_alloc) = 0;
       #endif
       return mat.memptr();
     }
   }
   
   size_t IO_GetParamColRows(const char* paramName)
   {
     return IO::GetParam<arma::vec>(paramName).n_rows;
   }
   
   double* IO_GetParamCol(const char* paramName)
   {
     // Are we using preallocated memory?  If so we have to handle this more
     // carefully.
     arma::vec& vec = IO::GetParam<arma::vec>(paramName);
     if (vec.n_elem <= arma::arma_config::mat_prealloc)
     {
       // Copy the memory to something we can give back to Julia.
       double* newMem = new double[vec.n_elem];
       arma::arrayops::copy(newMem, vec.mem, vec.n_elem);
       return newMem; // We believe Julia will free it.  Hopefully we are right.
     }
     else
     {
       arma::access::rw(vec.mem_state) = 1;
       #if ARMA_VERSION_MAJOR >= 10
         arma::access::rw(vec.n_alloc) = 0;
       #endif
       return vec.memptr();
     }
   }
   
   size_t IO_GetParamUColRows(const char* paramName)
   {
     return IO::GetParam<arma::Col<size_t>>(paramName).n_rows;
   }
   
   size_t* IO_GetParamUCol(const char* paramName)
   {
     arma::Col<size_t>& vec = IO::GetParam<arma::Col<size_t>>(paramName);
   
     // Are we using preallocated memory?  If so we have to handle this more
     // carefully.
     if (vec.n_elem <= arma::arma_config::mat_prealloc)
     {
       // Copy the memory to something we can give back to Julia.
       size_t* newMem = new size_t[vec.n_elem];
       arma::arrayops::copy(newMem, vec.mem, vec.n_elem);
       return newMem; // We believe Julia will free it.  Hopefully we are right.
     }
     else
     {
       arma::access::rw(vec.mem_state) = 1;
       #if ARMA_VERSION_MAJOR >= 10
         arma::access::rw(vec.n_alloc) = 0;
       #endif
       return vec.memptr();
     }
   }
   
   size_t IO_GetParamRowCols(const char* paramName)
   {
     return IO::GetParam<arma::rowvec>(paramName).n_cols;
   }
   
   double* IO_GetParamRow(const char* paramName)
   {
     // Are we using preallocated memory?  If so we have to handle this more
     // carefully.
     arma::rowvec& vec = IO::GetParam<arma::rowvec>(paramName);
     if (vec.n_elem <= arma::arma_config::mat_prealloc)
     {
       // Copy the memory to something we can give back to Julia.
       double* newMem = new double[vec.n_elem];
       arma::arrayops::copy(newMem, vec.mem, vec.n_elem);
       return newMem;
     }
     else
     {
       arma::access::rw(vec.mem_state) = 1;
       #if ARMA_VERSION_MAJOR >= 10
         arma::access::rw(vec.n_alloc) = 0;
       #endif
       return vec.memptr();
     }
   }
   
   size_t IO_GetParamURowCols(const char* paramName)
   {
     return IO::GetParam<arma::Row<size_t>>(paramName).n_cols;
   }
   
   size_t* IO_GetParamURow(const char* paramName)
   {
     arma::Row<size_t>& vec = IO::GetParam<arma::Row<size_t>>(paramName);
   
     // Are we using preallocated memory?  If so we have to handle this more
     // carefully.
     if (vec.n_elem <= arma::arma_config::mat_prealloc)
     {
       // Copy the memory to something we can give back to Julia.
       size_t* newMem = new size_t[vec.n_elem];
       arma::arrayops::copy(newMem, vec.mem, vec.n_elem);
       return newMem;
     }
     else
     {
       arma::access::rw(vec.mem_state) = 1;
       #if ARMA_VERSION_MAJOR >= 10
         arma::access::rw(vec.n_alloc) = 0;
       #endif
       return vec.memptr();
     }
   }
   
   size_t IO_GetParamMatWithInfoRows(const char* paramName)
   {
     return std::get<1>(IO::GetParam<std::tuple<data::DatasetInfo, arma::mat>>(
         paramName)).n_rows;
   }
   
   size_t IO_GetParamMatWithInfoCols(const char* paramName)
   {
     return std::get<1>(IO::GetParam<std::tuple<data::DatasetInfo, arma::mat>>(
         paramName)).n_cols;
   }
   
   bool* IO_GetParamMatWithInfoBoolPtr(const char* paramName)
   {
     const data::DatasetInfo& d = std::get<0>(
         IO::GetParam<std::tuple<data::DatasetInfo, arma::mat>>(paramName));
   
     bool* dims = new bool[d.Dimensionality()];
     for (size_t i = 0; i < d.Dimensionality(); ++i)
       dims[i] = (d.Type(i) == data::Datatype::numeric) ? false : true;
   
     return dims;
   }
   
   double* IO_GetParamMatWithInfoPtr(const char* paramName)
   {
     // Are we using preallocated memory?  If so we have to handle this more
     // carefully.
     arma::mat& m = std::get<1>(
         IO::GetParam<std::tuple<data::DatasetInfo, arma::mat>>(paramName));
     if (m.n_elem <= arma::arma_config::mat_prealloc)
     {
       double* newMem = new double[m.n_elem];
       arma::arrayops::copy(newMem, m.mem, m.n_elem);
       return newMem;
     }
     else
     {
       arma::access::rw(m.mem_state) = 1;
       #if ARMA_VERSION_MAJOR >= 10
         arma::access::rw(m.n_alloc) = 0;
       #endif
       return m.memptr();
     }
   }
   
   void IO_EnableVerbose()
   {
     Log::Info.ignoreInput = false;
   }
   
   void IO_DisableVerbose()
   {
     Log::Info.ignoreInput = true;
   }
   
   void IO_ResetTimers()
   {
     IO::GetSingleton().timer.Reset();
   }
   
   void IO_SetPassed(const char* paramName)
   {
     IO::SetPassed(paramName);
   }
   
   } // extern "C"
