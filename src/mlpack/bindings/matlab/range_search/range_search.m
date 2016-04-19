function result = range_search(dataPoints, maxDistance, varargin)
%Range Search
%
%  This function implements range search with a Euclidean distance metric. For a
%  given query point, a given range, and a given set of reference points, the
%  program will return all of the reference points with distance to the query
%  point in the given range.  This is performed for an entire set of query
%  points. You may specify a separate set of reference and query points, or only
%  a reference set -- which is then used as both the reference and query set.
%  The given range is taken to be inclusive (that is, points with a distance
%  exactly equal to the minimum and maximum of the range are included in the
%  results).
%
%  For example, the following will calculate the points within the range [2, 5]
%  of each point in 'input.csv' and store the distances in 'distances.csv' and
%  the neighbors in 'neighbors.csv':
%
%Parameters:
% dataPoints  - (required) Matrix containing the reference dataset.
% maxDistance - (required) The upper bound of the range.
% minDistance	- (optional) The lower bound. The default value is zero.
% queryPoints - (optional) Range search query points.
% leafSize    - (optional) Leaf size for tree building.  Default value 20.
% naive			  - (optional) If true, O(n^2) naive mode is used for computation.
% singleMode  - (optional) If true, single-tree search is used (as opposed to
%               dual-tree search.

% a parser for the inputs
p = inputParser;
p.addParamValue('minDistance', 0, @isscalar);
p.addParamValue('queryPoints', zeros(0), @ismatrix);
p.addParamValue('leafSize', 20, @isscalar);
p.addParamValue('naive', false, @(x) (x == true) || (x == false));
p.addParamValue('singleMode', false, @(x) (x == true) || (x == false));

% parsing the varargin options
p.parse(varargin{:});
parsed = p.Results;

% interfacing with mlpack
result = mex_range_search(dataPoints', maxDistance, ...
	parsed.minDistance, parsed.queryPoints', parsed.leafSize, ...
	parsed.naive, parsed.singleMode);




