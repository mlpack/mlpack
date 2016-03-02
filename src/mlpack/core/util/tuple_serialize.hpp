#ifndef MLPACK_CORE_SERIALIZE_TUPLE
#define MLPACK_CORE_SERIALIZE_TUPLE

#include <mlpack/core.hpp>

#include <string>

namespace boost {
namespace serialization {  
 
  template<
      size_t I, 
      size_t Max,
      typename Archive, 
      typename... Args
  >
  typename std::enable_if<I < Max, void>::type
  serialize(Archive& ar, std::tuple<Args...>& t, const unsigned int /* version */)
  {
    ar & data::CreateNVP(std::get<I>(t), "tuple" + std::to_string(I));
    serialize<I+1, Max>(ar, t, 0);
  }
  
  template<
      size_t I, 
      size_t Max,
      typename Archive, 
      typename... Args
  >
  typename std::enable_if<I == Max, void>::type 
  serialize(Archive&, std::tuple<Args...>&, const unsigned int /* version */)
  {    
  }

  template<typename Archive, typename... Args>
  void serialize(Archive& ar, std::tuple<Args...>& t, const unsigned int /* version */)
  {
    serialize<0, std::tuple_size<std::tuple<Args...>>::value-1>(ar, t, 0);
  }
}
}
#endif
