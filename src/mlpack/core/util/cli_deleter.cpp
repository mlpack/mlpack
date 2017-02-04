/**
 * @file cli_deleter.cpp
 * @author Ryan Curtin
 *
 * Extremely simple class whose only job is to delete the existing CLI object at
 * the end of execution.  This is meant to allow the user to avoid typing
 * 'CLI::Destroy()' at the end of their program.  The file also defines a static
 * CLIDeleter class, which will be initialized at the beginning of the program
 * and deleted at the end.  The destructor destroys the CLI singleton.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "cli_deleter.hpp"
#include "cli.hpp"

using namespace mlpack;
using namespace mlpack::util;

/***
 * Empty constructor that does nothing.
 */
CLIDeleter::CLIDeleter()
{
  /* Nothing to do. */
}

/***
 * This destructor deletes the CLI singleton.
 */
CLIDeleter::~CLIDeleter()
{
  // Delete the singleton!
  CLI::Destroy();
}
