function [est] = histogram_cv(D)

  [N, d] = size(D);
  if d == 1
    [S I] = sort(D);
    range = S(N) - S(1);
    min_dist = range + 1;
    for i = 1:(N-1)
      gap = S(i+1) - S(i);
      if gap < min_dist
	if gap > 0.0
	  min_dist = gap;
	end
      end
    end

    best_h = 0.0;
    h = min_dist;
    best_cv_score = inf;
    while h <= range
      m = ceil(range / h);
      B = hist(D, m);
      sum_pj_sq = (B * B') / (N^2);
      cv_score = (2  - ((N+1) * sum_pj_sq)) / (h * (N-1));
      if cv_score < best_cv_score
	best_h = h;
	best_cv_score = cv_score;
      end
      
      h = h * 5;
      m = ceil(range / h);
      B = hist(D, m);
      sum_pj_sq = (B * B') / (N^2);
      cv_score = (2  - ((N+1) * sum_pj_sq)) / (h * (N-1));
      if cv_score < best_cv_score
	best_h = h;
	best_cv_score = cv_score;
      end

      h = h * 2;
    end
    
% $$$   display(range);
% $$$   display(min_dist);
% $$$   display(best_cv_score);
    display(best_h);
    
    m = ceil(range / best_h);
    [B, X] = hist(D, m);
    
    for ind = 1:N
      bin_ind = -1;
      if D(ind) - X(1) < 0
	bin_ind = 1;
      else
	bin_ind = int16((D(ind) - X(1)) / best_h) + 1;
      end
      
      est(ind) = B(bin_ind) / (N * best_h);
    end
    
    points = 1;
    plot_cods(points, 1) = X(1) - (best_h / 2);
    plot_cods(points, 2) = 0.0;
    for ind = 1:m
      plot_cods(points + 1, 1) = X(ind) - (best_h / 2) + (min_dist/10);
      plot_cods(points + 1, 2) = B(ind) / (N * best_h);
      plot_cods(points + 2, 1) = X(ind) + (best_h / 2) - (min_dist/10);
      plot_cods(points + 2, 2) = B(ind) / (N * best_h);
      points = points + 2;
    end
    plot_cods(points + 1, 1) = X(m) + (best_h / 2);
    plot_cods(points + 1, 2) = 0.0;
    plot(plot_cods(:,1), plot_cods(:,2), '-k'); hold;
    xlabel('x', 'fontsize', 12);
    ylabel('density', 'fontsize', 12);
    
    
  else
    
    if d == 2
      S = sort(D);
      range_x = S(N,1) - S(1,1);
      range_y = S(N,2) - S(1,2);
      max_range =  max(range_x, range_x);
      min_dist = max_range + 1;
      for i = 1:(N-1)
	gap = S(i+1, 1) - S(i,1);
	if gap < min_dist
	  if gap > 0.0
	    min_dist = gap;
	  end
	end
	gap = S(i+1, 2) - S(i,2);
	if gap < min_dist
	  if gap > 0.0
	    min_dist = gap;
	  end
	end
      end

      display(max_range);
      display(min_dist);

      
      best_h = 0.0;
      h = max(min_dist, (max_range / N));
      best_cv_score = inf;
      while h <= max_range
	m_x = ceil(range_x / h);
	m_y = ceil(range_y / h);
	B = hist3(D, [m_x, m_y]);
	sum_pj_sq = sum(reshape((B .* B), size(B,1)*size(B,2), 1)) / (N^2);
	cv_score = (2  - ((N+1) * sum_pj_sq)) / (h^2 * (N-1));
	if cv_score < best_cv_score
	  best_h = h;
	  best_cv_score = cv_score;
	end
      
	h = h * 5;

	m_x = ceil(range_x / h);
	m_y = ceil(range_y / h);
	B = hist3(D, [m_x, m_y]);
	sum_pj_sq = sum(reshape((B .* B), size(B,1)*size(B,2), 1)) / (N^2);
	cv_score = (2  - ((N+1) * sum_pj_sq)) / (h^2 * (N-1));
	if cv_score < best_cv_score
	  best_h = h;
	  best_cv_score = cv_score;
	end

	h = h * 2;
      end
    
      display(best_cv_score);
      display(best_h);
    
      m_x = ceil(range_x / best_h);
      m_y = ceil(range_y / best_h);
      hist3(D, [m_x, m_y]);
      hold;
      grid off;
      xlabel('Relevant dimension', 'fontsize', 14);
      ylabel('Irrelevant dimension', 'fontsize', 14);
      zlabel('Density', 'fontsize', 14);
      Z = [0:10:40];
      Z = Z / (N * best_h^2);
      set(gca, 'ZTick', 0:10:40);
      set(gca, 'ZTickLabel', {Z(1), Z(2), Z(3), Z(4), Z(5)});
      hold;
      clear all;
    end
  end
  
  
