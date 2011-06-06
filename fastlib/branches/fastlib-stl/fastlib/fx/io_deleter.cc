/***
 * @file io_deleter.cc
 * @author Ryan Curtin
 *
 * Extremely simple class whose only job is to delete the existing IO object at
 * the end of execution.  This is meant to allow the user to avoid typing
 * 'IO::Destroy()' at the end of their program.  The file also defines a static
 * IODeleter class, which will be initialized at the beginning of the program
 * and deleted at the end.  The destructor destroys the IO singleton.
 */
#include "io_deleter.h"
#include "io.h"

using namespace mlpack;
using namespace mlpack::io;

/***
 * Empty constructor that does nothing.
 */
IODeleter::IODeleter() {
  /* nothing to do */
}

/***
 * This destructor deletes the IO singleton.
 */
IODeleter::~IODeleter() {
  // Delete the singleton!
  IO::Destroy();
}
