
function voteScore = voteScore(recommender, num, frame, period, target_sz, nRecommender,beta)
% Calculate votes of each recommender
   
   Fm_lib(period, nRecommender) = 0;
   for i = 1 : nRecommender 
      Fm_lib_org = calcF_measure( recommender(num).rect_position(frame - period + 1:frame,:) , recommender(i).rect_position(frame - period + 1:frame,:) );
      % calculate the F-measure
      Fm_lib(:,i) = exp(-(1 - Fm_lib_org).^2);    
      % a nonlinear gaussian function is used to transform the F-measure
   end
   
   const =[];
   weight = [];
   for j = 1:period
       TLE = 2 * sqrt( sum((recommender(num).center(frame-(period-j),:)-recommender(num).center(frame,:)).^2) )./ sqrt(sum((target_sz([2,1])).^2));
       const=[const exp(-TLE^2)];
       % calculating the consistency votes
       weight=[weight exp(-TLE)];
       % calculating the weights
   end
   
   % the average F-measure
   avgFm = sum(Fm_lib, 2)/nRecommender;           
   % the average F-measure for each recommender in a period for variance computation
   recAvgFm = sum(Fm_lib, 1)/period;        
   varAvgFm = sqrt( sum((Fm_lib - repmat(recAvgFm, period ,1) ).^2, 2)/nRecommender );  % the variance
   % temporal stability
   norm_factor = 1/sum(weight);
   weightAveFm = norm_factor*(weight*avgFm);
   weightVarFm = norm_factor*(weight*varAvgFm);
   agreeScore = weightAveFm./(weightVarFm+0.008); 
   constScore = norm_factor*sum(const.*weight);
   % total votes is the weighted sum of consistency and agreement votes
   voteScore = (1-beta) * agreeScore + beta * constScore;
end

