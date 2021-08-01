
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_python_mlpack_arma_util.hpp:

Program Listing for File arma_util.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_python_mlpack_arma_util.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/python/mlpack/arma_util.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_PYTHON_CYTHON_ARMA_UTIL_HPP
   #define MLPACK_BINDINGS_PYTHON_CYTHON_ARMA_UTIL_HPP
   
   // Include Armadillo via mlpack.
   #include <mlpack/core.hpp>
   
   template<typename T>
   void SetMemState(T& t, int state)
   {
     const_cast<arma::uhword&>(t.mem_state) = state;
     // If we just "released" the memory, so that the matrix does not own it, with
     // Armadillo 10 we must also ensure that the matrix does not deallocate the
     // memory by specifying `n_alloc = 0`.
     #if ARMA_VERSION_MAJOR >= 10
       const_cast<arma::uword&>(t.n_alloc) = 0;
     #endif
   }
   
   template<typename T>
   size_t GetMemState(T& t)
   {
     // Fake the memory state if we are using preallocated memory---since we will
     // end up copying that memory, NumPy can own it.
     if (t.mem && t.n_elem <= arma::arma_config::mat_prealloc)
       return 0;
   
     return (size_t) t.mem_state;
   }
   
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
       return m.memptr();
     }
   }
   
   #endif
