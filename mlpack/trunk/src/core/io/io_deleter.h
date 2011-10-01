/***
 * @file io_deleter.h
 * @author Ryan Curtin
 *
 * Extremely simple class whose only job is to delete the existing IO object at
 * the end of execution.  This is meant to allow the user to avoid typing
 * 'IO::Destroy()' at the end of their program.  The file also defines a static
 * IODeleter class, which will be initialized at the beginning of the program
 * and deleted at the end.  The destructor destroys the IO singleton.
 */
#ifndef __MLPACK_IO_IO_DELETER_H
#define __MLPACK_IO_IO_DELETER_H

namespace mlpack {
namespace io {

class IODeleter {
 public:
  IODeleter();
  ~IODeleter();
};

static IODeleter io_deleter;

}; // namespace io
}; // namespace mlpack

#endif
