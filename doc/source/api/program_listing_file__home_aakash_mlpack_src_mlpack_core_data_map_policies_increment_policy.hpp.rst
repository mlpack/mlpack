
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_data_map_policies_increment_policy.hpp:

Program Listing for File increment_policy.hpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_data_map_policies_increment_policy.hpp>` (``/home/aakash/mlpack/src/mlpack/core/data/map_policies/increment_policy.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DATA_MAP_POLICIES_INCREMENT_POLICY_HPP
   #define MLPACK_CORE_DATA_MAP_POLICIES_INCREMENT_POLICY_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <unordered_map>
   #include <mlpack/core/data/map_policies/datatype.hpp>
   
   namespace mlpack {
   namespace data {
   
   class IncrementPolicy
   {
    public:
     IncrementPolicy(const bool forceAllMappings = false) :
         forceAllMappings(forceAllMappings) { }
   
     // typedef of MappedType
     using MappedType = size_t;
   
     static const bool NeedsFirstPass = true;
   
     template<typename T, typename InputType>
     void MapFirstPass(const InputType& input,
                       const size_t dim,
                       std::vector<Datatype>& types)
     {
       if (types[dim] == Datatype::categorical)
       {
         // No need to check; it's already categorical.
         return;
       }
   
       if (forceAllMappings)
       {
         types[dim] = Datatype::categorical;
       }
       else
       {
         // Attempt to convert the input to an output type via a stringstream.
         std::stringstream token;
         token << input;
         T val;
         token >> val;
   
         if (token.fail() || !token.eof())
           types[dim] = Datatype::categorical;
       }
     }
   
     template<typename MapType, typename T, typename InputType>
     T MapString(const InputType& input,
                 const size_t dimension,
                 MapType& maps,
                 std::vector<Datatype>& types)
     {
       // If we are in a categorical dimension we already know we need to map.
       if (types[dimension] == Datatype::numeric && !forceAllMappings)
       {
         // Check if this input needs to be mapped or if it can be read
         // directly as a number.  This will be true if nothing else in this
         // dimension has yet been mapped, but this can't be read as a number.
         std::stringstream token;
         token << input;
         T val;
         token >> val;
   
         if (!token.fail() && token.eof())
           return val;
   
         // Otherwise, we must map.
       }
   
       // If this condition is true, either we have no mapping for the given input
       // or we have no mappings for the given dimension at all.  In either case,
       // we create a mapping.
       if (maps.count(dimension) == 0 ||
           maps[dimension].first.count(input) == 0)
       {
         // This input does not exist yet.
         size_t numMappings = maps[dimension].first.size();
   
         // Change type of the feature to categorical.
         if (numMappings == 0)
           types[dimension] = Datatype::categorical;
   
         typedef typename std::pair<InputType, MappedType> PairType;
         maps[dimension].first.insert(PairType(input, numMappings));
   
         // Do we need to create the second map?
         if (maps[dimension].second.count(numMappings) == 0)
         {
           maps[dimension].second.insert(std::make_pair(numMappings,
               std::vector<InputType>()));
         }
         maps[dimension].second[numMappings].push_back(input);
   
         return T(numMappings);
       }
       else
       {
         // This input already exists in the mapping.
         return maps[dimension].first.at(input);
       }
     }
   
    private:
     // Whether or not we should map all tokens.
     bool forceAllMappings;
   }; // class IncrementPolicy
   
   } // namespace data
   } // namespace mlpack
   
   #endif
