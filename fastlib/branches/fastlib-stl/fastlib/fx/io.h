#ifndef IO_H
#define IO_H

#include <boost/program_options.hpp>
#include <iostream>
#include <map>
#include <string>

#include "optionshierarchy.h"

namespace po = boost::program_options;

namespace mlpack {
	class IO {
		public:
			/* Adds a parameter to the heirarchy.  Incidentally, we are using char* and not std::string 
				since the vast majority of use cases will be literal strings*/
			static void add(const char* identifier, const char* description, const char* parent= NULL);
		
			/* If the argument requires a parameter, you must specify a type */
			template<class T>
			static void add(const char* identifier, const char* description, const char* parent = NULL) {
				//TODO: use singleton for state, wrap this up a parallel data structure 
				po::options_description& desc = getSingleton().desc;
	
				std::string path = getSingleton().manageHierarchy(identifier, parent, description);
	
				desc.add_options()
					(path.c_str(), po::value<T>(), description);
				return;
			}

		
			/* See if the specified flag was found while parsing.  Non-zero return value indicates success.*/
			static int checkValue(const char* identifier);
			
			/* Grab the value of type T found while parsing.  Non-zero return value indicates success.*/
			template<typename T>
			static T getValue(const char* identifier) {
				T tmp;
				if(checkValue(identifier))
					tmp = getSingleton().vmap[identifier].as<T>();
				return tmp;
			}
		
			/* The proper commandline parse method */
			static void parseCommandLine(int argc, char** argv);
		
			/* Parses a stream for options & values */
			static void parseStream(std::istream& stream);
		
			/* Prints out the current heirachy */
			static void print();
			
			
		private:
			/* Private member variables & methods */
		
			/* Hierarchy, everything is in a global namespace.  
				That said, this namespace will require qualified names.
				These names will take the form of node/child/child2 */
			
		
			//The documentation and names of options
			po::options_description desc;
		
			//Values of the options given by user
			po::variables_map vmap;
		
		
			//Store a relative index of path names
			OptionsHierarchy hierarchy;
		
			//Sanity checks strings before sending them to optionshierarchy
			//Returns the pathname placed in the hierarchy
			std::string manageHierarchy(const char* id, const char* parent, const char* description = "");
		
			//The singleton, obviously
			static IO* singleton;
		
			
			/* Not exposed to the outside, so as to spare users some ungainly
				x.getSingleton().foo() syntax 
				In this case, the singleton is used to store data for the static methods, 
				as there is no point in defining static methods only to have users call 
				private instance methods */
		
			/* Returns the singleton instance for use in the static methods */
			static IO& getSingleton();
		
			/* Make the constructor private, to preclude unauthorized instances */
			IO();
			//Initialize desc with a particular name
			IO(std::string& optionsName);
			IO(const IO& other);
			~IO();			
	};
};

#endif