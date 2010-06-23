
#include "tokenizer.h"

using namespace std;

void tokenizeString( const string& str, const string& delimiters,
    vector<string>& result, index_t pos,
    const string& stopon, index_t stopat, bool save_last ) {
  result.reserve(stopat);

  size_t last = pos;

  // When "unlimited" tokens desired, allow for numeric_limits<index_t>::max()
  if( !stopat )
    stopat = numeric_limits<index_t>::max();

  // loop through each character
  for( ; pos < str.length() && stopat; ++pos ) {
    // check that our current character is not a stopping character
    for( string::const_iterator stopit = stopon.begin();
        stopit < stopon.end(); ++stopit ) {
      if( str[pos] == *stopit ) {
        if( stopat && last < pos )
          result.push_back( str.substr( last, pos - last ) );
        if( save_last )
          result.push_back( str.substr( pos, str.size() - pos) );
        return;
      }
    }
    // check that our current character is not a delimiter
    for( size_t dpos = 0; dpos < delimiters.length(); ++dpos ) {
      if( str[pos] == delimiters[dpos] ) {
        if( last == pos )
          ++last;
        else {
          result.push_back( str.substr( last, pos - last ) );
          last = pos + 1;
          --stopat;
        }
        break;
      }
    }

  }

  // Grab the last token
  if( stopat && last < str.length() )
    result.push_back( str.substr( last, pos - last ) );
}

