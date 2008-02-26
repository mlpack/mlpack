for i = 1:size(ic_curves,2)
  scale_up_factor = ...
      1 / sqrt(sum(sum((ic_coef(:,i) * ic_coef(:,i)') .* ...
		       basis_inner_products)));
  ic_coef(:,i) = scale_up_factor * ic_coef(:,i);
  ic_curves(:,i) = scale_up_factor * ic_curves(:,i);
  ic_scores(i,:) = scale_up_factor * ic_scores(i,:);
end
