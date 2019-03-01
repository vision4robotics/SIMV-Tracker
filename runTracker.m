
clc;
clear all;
addpath('./function');                  
addpath('./utilization');
addpath('vgg','matconvnet/matlab');
vl_setupnn();
%%% Note that the default setting is CPU. TO ENABLE GPU, please recompile the MatConvNet toolbox  
% vl_compilenn('enableGpu',true);
% global enableGPU;
% enableGPU = true;
params.visualization = 1;                  % show output bbox on frame

%% load video info 
video_path_UAV123 = 'D:\SUN\Akamemie-学术\2019毕业设计\Dataset\UAV123_10fps\data_seq\UAV123_10fps';
ground_truth_path_UAV123 = 'D:\SUN\Akamemie-学术\2019毕业设计\Dataset\UAV123_10fps\anno\UAV123_10fps';
% please set the directory for your UAV123 image sequences.
video_name = choose_video(ground_truth_path_UAV123);
seq = load_video_info_UAV123(video_name, video_path_UAV123, ground_truth_path_UAV123);
video_path = seq.video_path;
ground_truth = seq.ground_truth;
videoname = video_name;
img_path = [video_path];

base_path = [video_path_UAV123 '\'];
img_files = seq.s_frames;
target_sz = [ground_truth(1,4), ground_truth(1,3)];
pos = [ground_truth(1,2), ground_truth(1,1)] + floor(target_sz/2);


% videoname = 'boat5'; 
% img_path = 'sequence/boat5/img/';
% base_path = 'sequence/';
% [img_files, pos, target_sz, video_path] = load_video_info(base_path, videoname);
% % you can also test the code on a single image sequence, i.e., boat5,if no UAV123
% % sequences is available for you
%% DSST related 
params.hog_cell_size = 4;
params.hog_scale_cell_size = 4;   
params.fixed_area = 200^2;                 % standard area to which we resize the target
params.n_bins = 2^5;                       % number of bins for the color histograms (bg and fg models)
params.lr_pwp_init = 0.01;                 % bg and fg color models learning rate 
params.inner_padding = 0.2;                % defines inner area used to sample colors from the foreground
params.output_sigma_factor = 0.1;          % standard deviation for the desired translation filter output 
params.lambda = 1e-4;                      % regularization weight
params.lr_cf_init = 0.01;                  % DSST learning rate
params.learning_rate_scale = 0.025;      
params.scale_sigma_factor = 1/2;
params.num_scales = 33;       
params.scale_model_factor = 1.0;
params.scale_step = 1.03;
params.scale_model_max_area = 32*16;
%% SIMV main parameters
params.period = 5;                       % frame period, \Delta k
params.update_thres = 2;                 % threshold for adaptive update
params.nRecommender = 7;                 % number of recommenders
params.beta = 0.99;                      % parameter for trading off the votes  
%% start SIMV_main.m
im = imread([img_path  img_files{1}]);

% is a grayscale sequence ?
if(size(im,3)==1)
    params.grayscale_sequence = true;
end
if(size(im,3)==3)
    params.grayscale_sequence = false;
end
params.img_files = img_files;
params.s_frames = img_files;
params.img_path = img_path;

% init_pos is the centre of the initial bounding box
params.init_pos = pos;
params.target_sz = target_sz;

[params, bg_area, fg_area, area_resize_factor] = initializeAllAreas(im, params);
if params.visualization
    params.videoPlayer = vision.VideoPlayer('Position', [100 100 [size(im,2), size(im,1)]+30]);
end
% start the actual tracking
results=SIMV_main(params, im, bg_area, fg_area, area_resize_factor);
positions = results.res;
fps = results.fps;

result_name = video_name;
SIMV = results;
savedir = './results/';
if ~exist(savedir,'dir')
    mkdir(savedir);
end   
save([savedir,result_name],'SIMV');
precision_plot(results.res,ground_truth,video_name, savedir,1);
% draw the precision score for SIMV on the sequence



