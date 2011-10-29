/**
 * @file Mat_extra_bones.hpp
 * @author Ryan Curtin
 *
 * Extra overload of load() and save() to allow transposition of matrix at load
 * time and save time.
 */

inline bool load(const std::string   name, const file_type type, const bool print_status, const bool transpose);

inline bool load(      std::istream& is,   const file_type type, const bool print_status, const bool transpose);

inline bool save(const std::string   name, const file_type type, const bool print_status, const bool transpose);

inline bool save(      std::ostream& os,   const file_type type, const bool print_status, const bool transpose);
