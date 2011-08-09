function [y_q, y_pq, T] = rtree(x, y, xranges, x_q, minSize, alpha)

T = Grow(x, y, xranges, minSize, alpha);
display(sprintf('Tree made...'));

[y_q, y_pq] = SearchTree(T, x_q);

% end rtree

function [T] = Grow(x, y, xranges, minSize, alpha)

[nDim, nPoints] = size(x);
T.size = nPoints;

[T.error, T.y] = NodeError(y, alpha);
% T.error = sum( (y - mean(y)).^2 );
T.mean_y = mean(y);
T.se_y = std(y);

if nPoints <= minSize
  T.is_leaf = 1;
  T.subtree_leaves = 1;
  T.subtree_error = T.error;
  %display(sprintf('L:%d - %d', nPoints, T.y));
else
  T.is_leaf = 0;
  
  [dim, sVal, red] = FindSplit(x, y, xranges, minSize, alpha);
  
  if red > 0.0
    
    if dim < 0
      display(sprintf('Something went terribly wrong'));
      return;
    end
    
    left_ind = find(x(dim, :) <= sVal);
    right_ind = find(x(dim, :) > sVal);

    % if (size(left_ind, 2) ~= 0 & size(right_ind, 2) ~= 0 )
    
    x_l = x(:, left_ind);
    y_l = y(left_ind);

    xranges_l = xranges;
    xranges_l(dim, 2) = sVal;


    x_r = x(:, right_ind);
    y_r = y(right_ind);

    xranges_r = xranges;
    xranges_r(dim, 1) = sVal;

    %display(sprintf('%d/%d - %d', size(x_l, 2), size(x_r, 2), T.y));
    
    T.left = Grow(x_l, y_l, xranges_l, minSize, alpha);
    T.right = Grow(x_r, y_r, xranges_r, minSize, alpha);
    
    T.subtree_leaves = T.left.subtree_leaves + ...
	T.right.subtree_leaves;
    
    T.subtree_error = T.left.subtree_error + T.right.subtree_error;
    
    T.dim = dim;
    T.sVal = sVal;
        
  else
    % make leaf if no split found
    
    T.is_leaf = 1;
    T.subtree_leaves = 1;
    T.subtree_error = T.error;
    %display(sprintf('L:%d - %d', nPoints, T.y));
  end
end

% end Grow


function [dim, sVal, max_red] = FindSplit(x, y, xranges, minSize, alpha)

[nDims, nPoints] = size(x);

%R_t = sum( (y - mean(y)).^2 );
[R_t, gam] = NodeError(y, alpha);

red = 0.0;
max_red = 0.0;

dim = -1;
sVal = -1.0;
lsize = 0;

num_dims_same_point = 0;

for i = 1:nDims

  % here you sort the xvalues such that for equal values of x, the
  % corresponding values of y s should be sorted as well because I
  % feel that the order of y for same values of x does affect the
  % partitioning + I feel all similar values of x should be on the
  % same side of the split - so need to ensure that as well. This
  % will avoid the ambiguity in the ordering of the ys.
  [x_sort ind] = sort(x(i, :));
  y_x_sort = y(ind);
  
  [x_uniq, m, n] = unique(x_sort);
  
  num_uniq_x = size(x_uniq, 2);
  
  if num_uniq_x > 1
    %display(sprintf('D:%d - ', i));
    for j = 1:num_uniq_x
      
      ind_l = find(x_sort <= x_uniq(j));
      ind_r = find(x_sort > x_uniq(j));
      
      [R_l, gam_l] = NodeError(y_x_sort(ind_l), alpha);
      [R_r, gam_r] = NodeError(y_x_sort(ind_r), alpha);
      
      %{R_l = sum((y(ind(1:j)) - mean(y(ind(1:j)))).^2);
      %R_r = sum((y(ind((j+1):nPoints)) - mean(y(ind((j+1): ...
      %					nPoints)))).^2);%}
      
      red = R_t - (R_l + R_r);
      if red == 0.0
	%display(sprintf('red:%f, L:%d, R:%d', red, gam_l, gam_r));
      end
      
      if red > max_red
	dim = i;
	max_red = red;
	lsize = size(ind_l, 2);
	sVal = x_uniq(j) ;
      end
      
    end % over the points
  else
    num_dims_same_point = num_dims_same_point + 1;   
  end % if all the same values in this dimension
  
end % over the dimensions


if (max_red > 0.0)
  %display(max_red);
  %display(dim);
  %display(lsize);
  %display(sprintf('Split found, %d/%d', lsize, (nPoints - lsize)));
else
  % display(sprintf('No split found - NDSP:%d/%d', num_dims_same_point, ...
	%	  nPoints));
end

% end FindSplit


function [error, gamma] = NodeError(y, alpha)

% For now it is the following using the max value
gamma = max(y);
error = sum( gamma - y );


% eventually we will try incorporating alpha if needed
%{
display(sprintf('Should be commented out...'));

t = 1;
while t == 0
g_max = max(y);

% E_q [gamma - |Uq|]_+
objFun = ObjFun(y, g_max);

% E_q [ I(|Uq| > gamma) ] 
constrFun = ConstrFun(y, g_max);

minObjFun = objFun;
minGamma = g_max;

while constrFun <= alpha
  
  if (minObjFun > objFun) 
    minObjFun = objFun;
    minGamma = g_max;
  end

  g_max = g_max - 1;
  objFun = ObjFun(y, g_max);
  constrFun = ConstrFun(y, g_max);
  
end

if g_max ~= minGamma - 1
  display(sprintf('Unexpected: %d -> %d', g_max, minGamma));
end

gamma = minGamma;
error = minObjFun;

end
%}

% end NodeError


function [obj] = ObjFun(data, thres)

indices = find(data <= thres);
n = size(data, 1);

obj = sum(thres - data(indices)); % / n;

% end ObjFun


function [constr] = ConstrFun(data, thres)

indices = find(data > thres);
n = size(data, 1);

constr = size(indices, 1) / n;

% end ConstrFun
