function [W, H] = nmf_l2(V, k);
% [W, H] = nmf_l2(V, W, H);

epsilon = eps;

W = rand([size(V,1) k]);
H = rand([k size(V,2)]);

converged = false;

epoch = 0;

while converged == false 
  
  epoch = epoch + 1;
  fprintf('epoch %d\n', epoch);
  
  W_old = W;
  H_old = H;
  
  WV = W' * V;
  WWH = W' * W * H;

  H = H .* WV ./ WWH;
  
  VH = V * H';
  WHH = W * H * H';
  
  W = W .* VH ./ WHH;
  
  if sum(sum(abs(W_old - W))) < epsilon
    if sum(sum(abs(H_old - H))) < epsilon
      converged = true;
    end
  end
end

V - W * H
