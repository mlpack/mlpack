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
 * This file is part of MLPACK 1.0.11.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
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
