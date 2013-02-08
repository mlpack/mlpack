function [distances, neighbors] = allkfn(dataPoints, k, varargin)
% [distances, neighbors] = allkfn(dataPoints, k, varargin)
%
% Calculate the all k-furthest-neighbors of a set of points.  You may specify a
% separate set of reference points and query points, or just a reference set
% which will be used as both the reference and query set.
%
% The output matrices are organized such that row i and column j in the
% neighbors matrix corresponds to the index of the point in the reference set
% which is the i'th furthest neighbor from the point in the query set with index
% j.  Row i and column j in the distances output matrix corresponds to the
% distance between those two points.
%
% Parameters:
%
% dataPoints - The reference set of data points.  Columns are assumed to
%              represent dimensions, with rows representing separate points.
% k          - The number of furthest neighbors to find.
%
% Optional parameters (i.e. allkfn(..., 'parameter', value, ...)):
%
% 'queryPoints' - An optional set of query points, if the reference and query
%                 sets are different.  Columns are assumed to represent
%                 dimensions, with rows representing separate points.
% 'leafSize'    - Leaf size in the kd-tree.  Defaults to 20.
% 'method'      - Algorithm to use.  'naive' uses naive O(n^2) computation;
%                 'single' uses single-tree traversal; 'dual' uses the standard
%                 dual-tree traversal.  Defaults to 'dual'.
%
% Examples:
%
% [distances, neighbors] = allkfn(dataPoints, 5);
% [distances, neighbors] = allkfn(dataPoints, 5, 'method', 'single');
% [distances, neighbors] = allkfn(dataPoints, 5, 'queryPoints', queryPoints);

% A parser for the inputs.
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

