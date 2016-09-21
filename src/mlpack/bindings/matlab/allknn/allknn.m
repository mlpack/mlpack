function [distances neighbors] = allknn(dataPoints, k, varargin)
%All K-Nearest-Neighbors
%
%  This program will calculate the all k-nearest-neighbors of a set of points
%  using kd-trees or cover trees (cover tree support is experimental and may not
%  be optimally fast). You may specify a separate set of reference points and
%  query points, or just a reference set which will be used as both the reference
%  and query set.
%
%  For example, the following will calculate the 5 nearest neighbors of eachpoint
%  in 'input.csv' and store the distances in 'distances.csv' and the neighbors in
%  the file 'neighbors.csv':

%  $ allknn --k=5 --reference_file=input.csv --distances_file=distances.csv
%    --neighbors_file=neighbors.csv

%  The output files are organized such that row i and column j in the neighbors
%  output file corresponds to the index of the point in the reference set which
%  is the i'th nearest neighbor from the point in the query set with index j.
%  Row i and column j in the distances output file corresponds to the distance
%  between those two points.
%
% Parameters:
% dataPoints - the matrix of data points. Columns are assumed to represent dimensions,
%              with rows representing seperate points.
% method     - the algorithm for computing the tree. 'naive' or 'boruvka', with
%              'boruvka' being the default algorithm.
% leafSize   - Leaf size in the kd-tree.  One-element leaves give the
%              empirically best performance, but at the cost of greater memory
%              requirements. One is default.
%
% Examples:
% result = emst(dataPoints);
% or
% esult = emst(dataPoints,'method','naive');

% a parser for the inputs
p = inputParser;
p.addParamValue('queryPoints', zeros(0), @ismatrix);
p.addParamValue('leafSize', 20, @isscalar);
p.addParamValue('naive', false, @(x) (x == true) || (x == false));
p.addParamValue('singleMode', false, @(x) (x == true) || (x == false));
p.addParamValue('coverTree', false, @(x) (x == true) || (x == false));

% parsing the varargin options
varargin{:}
p.parse(varargin{:});
parsed = p.Results;
parsed

% interfacing with mlpack
[distances neighbors] = mex_allknn(dataPoints', k, parsed.queryPoints', ...
	parsed.leafSize, parsed.naive, parsed.singleMode, parsed.coverTree);

% transposing results
distances = distances';
neighbors = neighbors' + 1; % matlab indices began at 1, not zero

return;

