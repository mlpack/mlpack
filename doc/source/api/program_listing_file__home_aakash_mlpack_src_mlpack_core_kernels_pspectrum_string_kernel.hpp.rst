
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_kernels_pspectrum_string_kernel.hpp:

Program Listing for File pspectrum_string_kernel.hpp
====================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_kernels_pspectrum_string_kernel.hpp>` (``/home/aakash/mlpack/src/mlpack/core/kernels/pspectrum_string_kernel.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_KERNELS_PSPECTRUM_STRING_KERNEL_HPP
   #define MLPACK_CORE_KERNELS_PSPECTRUM_STRING_KERNEL_HPP
   
   #include <map>
   #include <string>
   #include <vector>
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/util/log.hpp>
   
   namespace mlpack {
   namespace kernel {
   
   class PSpectrumStringKernel
   {
    public:
     PSpectrumStringKernel(const std::vector<std::vector<std::string> >& datasets,
                           const size_t p);
   
     template<typename VecType>
     double Evaluate(const VecType& a, const VecType& b) const;
   
     const std::vector<std::vector<std::map<std::string, int> > >& Counts() const
     { return counts; }
     std::vector<std::vector<std::map<std::string, int> > >& Counts()
     { return counts; }
   
     size_t P() const { return p; }
     size_t& P() { return p; }
   
    private:
     std::vector<std::vector<std::map<std::string, int> > > counts;
   
     size_t p;
   };
   
   } // namespace kernel
   } // namespace mlpack
   
   // Include implementation of templated Evaluate().
   #include "pspectrum_string_kernel_impl.hpp"
   
   #endif
