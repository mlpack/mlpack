# store output of cpplint in bash variable
cpplintOut=`python cpplint.py --extensions=hpp,cpp --filter=\
-whitespace/braces,\
-whitespace/newline,\
-build/header_guard,\
-build/include_order,\
-build/storage_class,\
-build/namespaces,\
-build/include_what_you_use,\
-legal/copyright,\
-readability/casting,\
-readability/alt_tokens,\
-readability/todo,\
-readability/multiline_string,\
-runtime/explicit,\
-runtime/int,\
-runtime/references \
$(find ./src/mlpack/core -iname '*.[hc]pp' -type f ! -path "./src/mlpack/core/arma_extend/*" ! -path "./src/mlpack/core/boost_backport/*") | \
grep -v 'Missing spaces around <'`

# find number of style errors
numErrors=$(echo "$cpplintOut" | grep "Total errors found:" | cut -d ':' -f 2)
echo "Total errors found:$numErrors"

# if errors > 0, style check fails else pass
if [ "$numErrors" -gt 0 ] 
then 
  echo "Style check failed."
  exit 1
fi
echo "Style check passed."
