function F_measure_orig = calcF_measure(recommender_m,recommender_n)

%each row is a rectangle.
% recommender_m(i,:) = [x y w h]
% recommender_n(j,:) = [x y w h]
% (x,y) is the bottom-left point of the rectangle
% (w,h) is the width and height of the rectangle


left_m = recommender_m(:,1);
bottom_m = recommender_m(:,2);
right_m = left_m + recommender_m(:,3) - 1;
top_m = bottom_m + recommender_m(:,4) - 1;
% (left_m, bottom_m) is the bottom-left point of the rectangle m
% (right_m, top_m) is the top_right point of the rectangle m

left_n = recommender_n(:,1);
bottom_n = recommender_n(:,2);
right_n = left_n + recommender_n(:,3) - 1;
top_n = bottom_n + recommender_n(:,4) - 1;
% (left_n, bottom_n) is the bottom-left point of the rectangle n
% (right_n, top_n) is the top_right point of the rectangle n

MN = (max(0, min(right_m, right_n) - max(left_m, left_n)+1 )) .* (max(0, min(top_m, top_n) - max(bottom_m, bottom_n)+1 ));
% calculate the area of the overlap region MN
area_m = recommender_m(:,3) .* recommender_m(:,4);
% calculate the area of the rectangle m
area_n = recommender_n(:,3) .* recommender_n(:,4);
% calculate the area of the rectangle n
F_measure_orig = 2.*MN./(area_m+area_n);
