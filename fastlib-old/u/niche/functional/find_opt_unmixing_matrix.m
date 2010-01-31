% find_opt_unmixing_matrix(X) - find the unmixing matrix of W that
% best separates the components of X
% X is a p x n matrix
function [Y_pos,Y_neg,W_pos,W_neg] = find_opt_unmixing_matrix(X);

p = size(X, 1);

current_X = X;

total_rotator = eye(p);

for epoch = 1:1

  for i = 1:p
    for j = i+1:p
      
      disp(sprintf('[i,j] = [%d,%d]', i, j));
      
      
      subspace = [current_X(i,:) ; current_X(j,:)];
      [theta_star, rotator_star] = find_opt_subrotation(subspace);
      
      new_rotator=eye(p);
      new_rotator(i,i)=cos(theta_star);
      new_rotator(i,j)=-sin(theta_star);
      new_rotator(j,i)=sin(theta_star);
      new_rotator(j,j)=cos(theta_star);
      
      total_rotator = new_rotator * total_rotator;
      current_X = total_rotator * X;
      
    end  
  end
end

Y_pos = current_X;

neg_total_rotator=eye(p);
neg_total_rotator(1,1)=cos(theta_star + pi);
neg_total_rotator(1,2)=-sin(theta_star + pi);
neg_total_rotator(2,1)=sin(theta_star + pi);
neg_total_rotator(2,2)=cos(theta_star + pi);

Y_neg = neg_total_rotator * X;
W_pos = total_rotator;
W_neg = neg_total_rotator;

