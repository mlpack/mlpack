function [y, y_p] = SearchTree(T, x, limit)

[nDim, nPoints] = size(x);
if nargin < 3
  limit = 0;
end


for i = 1:nPoints
  [y(i), y_p(i)] = FindY(T, x(:,i), limit);
end


% end SearchTree


function [y, y_p] = FindY(T, x, limit)

if T.is_leaf == 1 | T.size <= limit
  y = T.y;
  y_p = T.mean_y + 2*T.se_y;
else
  if x(T.dim) <= T.sVal
    [y, y_p] = FindY(T.left, x);
  else
    [y, y_p] = FindY(T.right, x);
  end
end

% end FindY
