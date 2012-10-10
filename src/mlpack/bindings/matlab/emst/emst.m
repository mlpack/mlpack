function result = emst(dataPoints, varargin)
% Fast Euclidean Minimum Spanning Tree. This script can compute
% the Euclidean minimum spanning tree of a set of input points using the 
% dual-tree Boruvka algorithm.
% 
% The output is saved in a three-column matrix, where each row indicates an
% edge.  The first column corresponds to the lesser index of the edge; the
% second column corresponds to the greater index of the edge; and the third
% column corresponds to the distance between the two points.
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
p.addOptional('method', 'boruvka', @(x) strcmpi(x, 'naive') || strcmpi(x, 'boruvka'));
p.addOptional('leafSize', 1, @isscalar);

% parsing the varargin options
p.parse(varargin{:});
parsed = p.Results;

% interfacing with mlpack. transposing to machine learning standards. 
if strcmpi(parsed.method, 'boruvka')
  result = mex_emst(dataPoints', 1, parsed.leafSize);
	result = result';
  return;
else
  result = mex_emst(dataPoints', 0, 1);
	result = result';
  return;
end

