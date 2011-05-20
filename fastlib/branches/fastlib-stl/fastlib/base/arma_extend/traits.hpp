// Extra traits to support u64 and s64 until that patch is applied to the
// Armadillo sources.

template<> struct isnt_supported_elem_type< u64 >                  : public isnt_supported_elem_type_false {};
template<> struct isnt_supported_elem_type< s64 >                  : public isnt_supported_elem_type_false {};

