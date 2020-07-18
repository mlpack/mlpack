#ifndef CEREAL_ARRAY_WRAPPER_HPP
#define CEREAL_ARRAY_WRAPPER_HPP

// This file add make_array functionality to cereal
// This functionality exist only in boost::serialization.
// Most part of this code are copied from array_wrapper in boost::serialization

#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/archives/json.hpp>

namespace cereal {

template<class T>
class array_wrapper
{
private:
    array_wrapper& operator=(array_wrapper rhs);
    // note: I would like to make the copy constructor private but this breaks
    // make_array.  So I make make_array a friend
    template<class Tx, class S>
    friend const cereal::array_wrapper<Tx> make_array(Tx * t, S s);
public:

    // array_wrapper(array_wrapper& rhs) :
    //     m_t(rhs.m_t),
    //     m_element_count(rhs.m_element_count)
    // {}
    array_wrapper(T * t, std::size_t s) :
        m_t(t),
        m_element_count(s)
    {}

    // default implementation
    // Cereal does not require to split member, it can do that internally
    // If this is the case we can not implement optimized version, since 
    // the only possible one is optimized.
    // Some verification needed...
    // default implementation
    template<class Archive>
    void serialize(Archive &ar, const unsigned int /*version*/)
    {
     // default implemention does the loop
      std::size_t c = count();
      T * t = address();
      while(0 < c--)
            ar & cereal::make_nvp("item", *t++); 
    }

    T * address() const
    {
      return m_t;
    }

    std::size_t count() const
    {
      return m_element_count;
    }

private:
    T * const m_t;
    const size_t m_element_count;
};

template<class T, class S>
inline
array_wrapper< T > make_array(T* t, S s){
    array_wrapper< T > a(t, s);
    return a;
}

} // end namespace cereal

#endif //CEREAL_ARRAY_WRAPPER_HPP
