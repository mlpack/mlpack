
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_go_mlpack_capi_arma_util.hpp:

Program Listing for File arma_util.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_go_mlpack_capi_arma_util.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/go/mlpack/capi/arma_util.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_GO_GONUM_ARMA_UTIL_HPP
   #define MLPACK_BINDINGS_GO_GONUM_ARMA_UTIL_HPP
   
   // Include Armadillo via mlpack.
   #include <mlpack/core/util/io.hpp>
   #include <mlpack/core.hpp>
   
   namespace mlpack {
   
   template<typename T>
   inline typename T::elem_type* GetMemory(T& m)
   {
     if (m.mem && m.n_elem <= arma::arma_config::mat_prealloc)
     {
       // We need to allocate new memory.
       typename T::elem_type* mem =
           arma::memory::acquire<typename T::elem_type>(m.n_elem);
       arma::arrayops::copy(mem, m.memptr(), m.n_elem);
       return mem;
     }
     else
     {
       arma::access::rw(m.mem_state) = 1;
       // With Armadillo 10 and newer, we must set `n_alloc` to 0 so that
       // Armadillo does not deallocate the memory.
       #if ARMA_VERSION_MAJOR >= 10
         arma::access::rw(m.n_alloc) = 0;
       #endif
       return m.memptr();
     }
   }
   
   } // namespace mlpack
   
   #endif
