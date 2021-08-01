
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_python_mlpack_io_util.hpp:

Program Listing for File io_util.hpp
====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_python_mlpack_io_util.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/python/mlpack/io_util.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_PYTHON_CYTHON_IO_UTIL_HPP
   #define MLPACK_BINDINGS_PYTHON_CYTHON_IO_UTIL_HPP
   
   #include <mlpack/core/util/io.hpp>
   #include <mlpack/core/data/dataset_mapper.hpp>
   
   namespace mlpack {
   namespace util {
   
   template<typename T>
   inline void SetParam(const std::string& identifier, T& value)
   {
     IO::GetParam<T>(identifier) = std::move(value);
   }
   
   template<typename T>
   inline void SetParamPtr(const std::string& identifier,
                           T* value,
                           const bool copy)
   {
     IO::GetParam<T*>(identifier) = copy ? new T(*value) : value;
   }
   
   template<typename T>
   inline void SetParamWithInfo(const std::string& identifier,
                                T& matrix,
                                const bool* dims)
   {
     typedef typename std::tuple<data::DatasetInfo, T> TupleType;
     typedef typename T::elem_type eT;
   
     // The true type of the parameter is std::tuple<T, DatasetInfo>.
     const size_t dimensions = matrix.n_rows;
     std::get<1>(IO::GetParam<TupleType>(identifier)) = std::move(matrix);
     data::DatasetInfo& di = std::get<0>(IO::GetParam<TupleType>(identifier));
     di = data::DatasetInfo(dimensions);
   
     bool hasCategoricals = false;
     for (size_t i = 0; i < dimensions; ++i)
     {
       if (dims[i])
       {
         di.Type(i) = data::Datatype::categorical;
         hasCategoricals = true;
       }
     }
   
     // Do we need to find how many categories we have?
     if (hasCategoricals)
     {
       arma::vec maxs = arma::max(
           std::get<1>(IO::GetParam<TupleType>(identifier)), 1);
   
       for (size_t i = 0; i < dimensions; ++i)
       {
         if (dims[i])
         {
           // Map the right number of objects.
           for (size_t j = 0; j < (size_t) maxs[i]; ++j)
           {
             std::ostringstream oss;
             oss << j;
             di.MapString<eT>(oss.str(), i);
           }
         }
       }
     }
   }
   
   template<typename T>
   T* GetParamPtr(const std::string& paramName)
   {
     return IO::GetParam<T*>(paramName);
   }
   
   template<typename T>
   T& GetParamWithInfo(const std::string& paramName)
   {
     // T will be the Armadillo type.
     typedef std::tuple<data::DatasetInfo, T> TupleType;
     return std::get<1>(IO::GetParam<TupleType>(paramName));
   }
   
   inline void EnableVerbose()
   {
     Log::Info.ignoreInput = false;
   }
   
   inline void DisableVerbose()
   {
     Log::Info.ignoreInput = true;
   }
   
   inline void DisableBacktrace()
   {
     Log::Fatal.backtrace = false;
   }
   
   inline void ResetTimers()
   {
     // Just get a new object---removes all old timers.
     IO::GetSingleton().timer.Reset();
   }
   
   inline void EnableTimers()
   {
     Timer::EnableTiming();
   }
   
   } // namespace util
   } // namespace mlpack
   
   #endif
