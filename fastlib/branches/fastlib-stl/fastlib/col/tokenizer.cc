
#include "tokenizer.h"

void tokenizeString( const std::string& str, const std::string& delimeters,
    std::vector<std::string>& result, index_t pos,
    const std::string& stopon, index_t stopat ) {
  result.reserve(stopat);

  size_t last = pos;

  // Allow for numeric_limits<index_t>::max() tokens
  // Minimum is INT_MAX (sufficient?)
  if( !stopat )
    --stopat;
  for( ; pos < str.length() && stopat; ++pos ) {

    for( std::string::const_iterator stopit = stopon.begin();
        stopit < stopon.end(); ++stopit ) {
      if( str[pos] == *stopit ) {
        if( stopat && last < pos )
          result.push_back(
              str.substr( last, pos - last )
              );
        return;
      }
    }
    
    for( size_t dpos = 0; dpos < delimeters.length(); ++dpos ) {
      if( str[pos] == delimeters[dpos] ) {
        if( last == pos )
          ++last;
        else {
          result.push_back( 
              str.substr( last, pos - last )
              );
          last = pos+1;
          --stopat;
        }
        break;
      }
    }

  }

  // Grab the last token
  if( stopat && last < str.length() )
    result.push_back(
        str.substr( last, pos - last )
        );
}

