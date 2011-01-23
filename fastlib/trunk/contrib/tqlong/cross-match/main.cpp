/*
 *      main.cpp
 *
 *      Copyright 2011 Long <tqlong@cs.etintin.com>
 *
 *      This program is free software; you can redistribute it and/or modify
 *      it under the terms of the GNU General Public License as published by
 *      the Free Software Foundation; either version 2 of the License, or
 *      (at your option) any later version.
 *
 *      This program is distributed in the hope that it will be useful,
 *      but WITHOUT ANY WARRANTY; without even the implied warranty of
 *      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *      GNU General Public License for more details.
 *
 *      You should have received a copy of the GNU General Public License
 *      along with this program; if not, write to the Free Software
 *      Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 *      MA 02110-1301, USA.
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <boost/program_options.hpp>
using namespace std;

boost::program_options::options_description desc("Allowed options");
boost::program_options::variables_map vm;

void process_options(int argc, char** argv)
{
  desc.add_options()
      ("help", "produce help message")
      ("in", boost::program_options::value<string>()->default_value("dr7mode1radec-150901200.bin") ,"file consists of text in original language")
      ("out", boost::program_options::value<string>()->default_value("out.txt") ,"file consists of being translated text")
      ("lin", boost::program_options::value<string>()->default_value("en") ,"original language")
      ("lout", boost::program_options::value<string>()->default_value("vi") , "destination language")
      ;

  boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
  boost::program_options::notify(vm);

  if (vm.count("help"))
  {
    cout << desc << endl;
    exit(1);
  }
}

struct ObjectRecord
{
  long long id;
  double ra, dec;
};

void readObjectFile(const string& filename)
{
  ifstream ifs(filename.c_str(), ios::in | ios::binary);
  if (!ifs.is_open())
  {
    cout << "Cannot open " << filename << "\n";
    return;
  }
  // get length of file:
  ifs.seekg (0, ios::end);
  long long length = ifs.tellg();
  ifs.seekg (0, ios::beg);
//  ifs.seekg (-(long long)100 * sizeof(ObjectRecord), ios::end);
  cout << "length = " << length << " "
       << "total = " << length / sizeof(ObjectRecord) << "\n";

  const int BUF_SIZE = 40000;
  ObjectRecord obj[BUF_SIZE];
  long long total = 0;
  while (ifs.good())
  {
    ifs.read((char*) &obj, sizeof(obj));
    long long count = ifs.gcount() / sizeof(ObjectRecord);
    total += count;
    cout << count << " " << total << "\n";
//    for (int i = 0; i < count; i++)
//      cout << setw(12) << obj[i].id
//           << setw(20) << obj[i].ra
//           << setw(20) << obj[i].dec
//           << "\n";
  }
  ifs.close();
}

int main(int argc, char** argv)
{
  process_options(argc, argv);
  cout << "cross-match processing...\n"
      << "object size = " << sizeof(ObjectRecord) << "\n";

  readObjectFile(vm["in"].as<string>());

  return 0;
}

