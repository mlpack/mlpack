/**
 * @file cli_deleter.cpp
 * @author Ryan Curtin
 *
 * Extremely simple class whose only job is to delete the existing CLI object at
 * the end of execution.  This is meant to allow the user to avoid typing
 * 'CLI::Destroy()' at the end of their program.  The file also defines a static
 * CLIDeleter class, which will be initialized at the beginning of the program
 * and deleted at the end.  The destructor destroys the CLI singleton.
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
