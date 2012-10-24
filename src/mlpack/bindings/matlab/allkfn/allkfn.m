function [distances neighbors] = allkfn(dataPoints, k, varargin)
%All K-Furthest-Neighbors
%
%  This program will calculate the all k-furthest-neighbors of a set of points.
%  You may specify a separate set of reference points and query points, or just a
%  reference set which will be used as both the reference and query set.
%  
%  For example, the following will calculate the 5 furthest neighbors of
%  eachpoint in 'input.csv' and store the distances in 'distances.csv' and the
%  neighbors in the file 'neighbors.csv':
%  
%  $ allkfn --k=5 --reference_file=input.csv --distances_file=distances.csv
%    --neighbors_file=neighbors.csv
%  
%  The output files are organized such that row i and column j in the neighbors
%  output file corresponds to the index of the point in the reference set which
%  is the i'th furthest neighbor from the point in the query set with index j. 
%  Row i and column j in the distances output file corresponds to the distance
%  between those two points.

% a parser for the inputs
p = inputParser;
p.addParamValue('queryPoints', zeros(0), @ismatrix);
p.addParamValue('leafSize', 20, @isscalar);
p.addParamValue('naive', false, @(x) (x == true) || (x == false));
p.addParamValue('singleMode', false, @(x) (x == true) || (x == false));

% parsing the varargin options
varargin{:}
p.parse(varargin{:});
parsed = p.Results;
parsed

% interfacing with mlpack
[distances neighbors] = mex_allkfn(dataPoints', k, parsed.queryPoints', ...
	parsed.leafSize, parsed.naive, parsed.singleMode);

% transposing results
distances = distances';
neighbors = neighbors' + 1; % matlab indices began at 1, not zero

return;

