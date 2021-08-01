
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_cereal_pair_associative_container.hpp:

Program Listing for File pair_associative_container.hpp
=======================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_cereal_pair_associative_container.hpp>` (``/home/aakash/mlpack/src/mlpack/core/cereal/pair_associative_container.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   /*
     Copyright (c) 2014, Randolph Voorhies, Shane Grant
     All rights reserved.
   
     Redistribution and use in source and binary forms, with or without
     modification, are permitted provided that the following conditions are met:
         * Redistributions of source code must retain the above copyright
           notice, this list of conditions and the following disclaimer.
         * Redistributions in binary form must reproduce the above copyright
           notice, this list of conditions and the following disclaimer in the
           documentation and/or other materials provided with the distribution.
         * Neither the name of cereal nor the
           names of its contributors may be used to endorse or promote products
           derived from this software without specific prior written permission.
   
     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
     ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
     WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
     DISCLAIMED. IN NO EVENT SHALL RANDOLPH VOORHIES OR SHANE GRANT BE LIABLE FOR ANY
     DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
     (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
     LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
     ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
     (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
     SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
   */
   #ifndef CEREAL_CONCEPTS_PAIR_ASSOCIATIVE_CONTAINER_HPP_
   #define CEREAL_CONCEPTS_PAIR_ASSOCIATIVE_CONTAINER_HPP_
   
   #include "cereal/cereal.hpp"
   
   namespace cereal
   {
     template <class Archive, template <typename...> class Map, typename... Args, typename = typename Map<Args...>::mapped_type> inline
     void CEREAL_SAVE_FUNCTION_NAME( Archive & ar, Map<Args...> const & map )
     {
       ar( make_size_tag( static_cast<size_type>(map.size()) ) );
   
       for( const auto & i : map )
         ar( make_map_item(i.first, i.second) );
     }
   
     template <class Archive, template <typename...> class Map, typename... Args, typename = typename Map<Args...>::mapped_type> inline
     void CEREAL_LOAD_FUNCTION_NAME( Archive & ar, Map<Args...> & map )
     {
       size_type size;
       ar( make_size_tag( size ) );
   
       map.clear();
   
       auto hint = map.begin();
       for( size_t i = 0; i < size; ++i )
       {
         typename Map<Args...>::key_type key;
         typename Map<Args...>::mapped_type value;
   
         ar( make_map_item(key, value) );
         #ifdef CEREAL_OLDER_GCC
         hint = map.insert( hint, std::make_pair(std::move(key), std::move(value)) );
         #else // NOT CEREAL_OLDER_GCC
         hint = map.emplace_hint( hint, std::move( key ), std::move( value ) );
         #endif // NOT CEREAL_OLDER_GCC
       }
     }
   } // namespace cereal
   
   #endif // CEREAL_CONCEPTS_PAIR_ASSOCIATIVE_CONTAINER_HPP_
