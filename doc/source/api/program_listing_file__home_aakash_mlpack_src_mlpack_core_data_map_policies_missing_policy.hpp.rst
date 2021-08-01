
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_data_map_policies_missing_policy.hpp:

Program Listing for File missing_policy.hpp
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_data_map_policies_missing_policy.hpp>` (``/home/aakash/mlpack/src/mlpack/core/data/map_policies/missing_policy.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DATA_MAP_POLICIES_MISSING_POLICY_HPP
   #define MLPACK_CORE_DATA_MAP_POLICIES_MISSING_POLICY_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <unordered_map>
   #include <mlpack/core/data/map_policies/datatype.hpp>
   #include <limits>
   #include <set>
   
   namespace mlpack {
   namespace data {
   
   class MissingPolicy
   {
    public:
     // typedef of MappedType
     using MappedType = double;
   
     MissingPolicy()
     {
       // Nothing to initialize here.
     }
   
     explicit MissingPolicy(std::set<std::string> missingSet) :
         missingSet(std::move(missingSet))
     {
       // Nothing to initialize here.
     }
   
     static const bool NeedsFirstPass = false;
   
     template<typename T>
     void MapFirstPass(const std::string& /* string */, const size_t /* dim */)
     {
       // Nothing to do.
     }
   
     template<typename MapType, typename T>
     T MapString(const std::string& string,
                 const size_t dimension,
                 MapType& maps,
                 std::vector<Datatype>& /* types */)
     {
       static_assert(std::numeric_limits<T>::has_quiet_NaN == true,
           "Cannot use MissingPolicy with types where has_quiet_NaN() is false!");
   
       // If we can load the string then there is no need for mapping.
       std::stringstream token;
       token.str(string);
       T t;
       token >> t; // Could be sped up by only doing this if we need to.
   
       MappedType value = std::numeric_limits<MappedType>::quiet_NaN();
       // But we can't use that for the map, so we need some other thing that will
       // represent quiet_NaN().
       const MappedType mapValue = std::nexttoward(
           std::numeric_limits<MappedType>::max(), MappedType(0));
   
       // If extraction of the value fails, or if it is a value that is supposed to
       // be mapped, then do mapping.
       if (token.fail() || !token.eof() ||
           missingSet.find(string) != std::end(missingSet))
       {
         // Everything is mapped to NaN.  However we must still keep track of
         // everything that we have mapped, so we add it to the maps if needed.
         if (maps.count(dimension) == 0 ||
             maps[dimension].first.count(string) == 0)
         {
           // This string does not exist yet.
           typedef std::pair<std::string, MappedType> PairType;
           maps[dimension].first.insert(PairType(string, value));
   
           // Insert right mapping too.
           if (maps[dimension].second.count(mapValue) == 0)
           {
             // Create new element in reverse map.
             maps[dimension].second.insert(std::make_pair(mapValue,
                 std::vector<std::string>()));
           }
           maps[dimension].second[mapValue].push_back(string);
         }
   
         return value;
       }
       else
       {
         // We can just return the value that we read.
         return t;
       }
     }
   
    private:
     // Note that missingSet and maps are different.
     // missingSet specifies which value/string should be mapped and may be a
     // superset of 'maps'.
     std::set<std::string> missingSet;
   }; // class MissingPolicy
   
   } // namespace data
   } // namespace mlpack
   
   #endif
