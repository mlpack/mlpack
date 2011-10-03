function [] = tree_purity(T, data, level)

if T.leaf == 1
  %display(sprintf('%s', level));
else
  [nR, nC] = size(data);

  Y(:,1) = data(:, nC);
  Y(:,2) = sign(data(:, 1:(nC-1)) * T.omega + T.b);

  wsum = 0;
  w = 0;

  for i = 1:10
    ind = find(Y(:,1) == i-1);
    signs = Y(ind, 2);
    A = size(signs, 1);
    if A > 0
      B = sum(signs);
      %x = (A+B) / 2;
      %y = (A-B) / 2;
      %Labels(i,1) = x; Labels(i,2) = y; Labels(i,3) = abs(B) / A;
      wsum = wsum + abs(B);
      w = w + A;
    end
    ind = [];
    signs = [];
  end

  display(sprintf('%s%0.3f:%d', level, (wsum / w), nR));
  ind = find(Y(:,2) == -1);
  data_l = data(ind, :);
  tree_purity(T.left, data_l, strcat(level, '-'));
  ind = [];
  ind = find(Y(:,2) == 1);
  data_r = data(ind, :);
  tree_purity(T.right, data_r, strcat(level, '-'));
end
