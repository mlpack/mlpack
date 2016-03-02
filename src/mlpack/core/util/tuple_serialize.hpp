#ifndef MLPACK_CORE_SERIALIZE_TUPLE
#define MLPACK_CORE_SERIALIZE_TUPLE

#include <mlpack/core.hpp>

#include <string>

namespace mlpack {
namespace serialization {  
 
  template<
      size_t I, 
      size_t Max,
      typename Archive, 
      typename... Args
  >
  typename std::enable_if<I < Max, void>::type
  Serialize(Archive& ar, std::tuple<Args...>& t, const unsigned int /* version */)
  {
    ar & data::CreateNVP(std::get<I>(t), "tuple" + std::to_string(I));
    Serialize<I+1, Max>(ar, t, 0);
  }
  
  template<
      size_t I, 
      size_t Max,
      typename Archive, 
      typename... Args
  >
  typename std::enable_if<I == Max, void>::type 
  Serialize(Archive&, std::tuple<Args...>&, const unsigned int /* version */)
  {    
  }       
}
}
#endif
