function demo()
% Please cite our work if this code helps you:
%@InProceedings{G_2016_CVPR,
%author = {Kumar B G, Vijay and Carneiro, Gustavo and Reid, Ian},
%title = {Learning Local Image Descriptors With Deep Siamese and Triplet Convolutional Networks by Minimising Global Loss Functions},
%booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
%month = {June},
%year = {2016}
%} 
root_path = pwd;
addpath(genpath(fullfile(root_path,'code')));
addpath(genpath(fullfile(root_path,'data')));
addpath(genpath(fullfile(root_path,'models')));
% run('Your matconvnet root path/matconvnet-1.0-beta13/matlab/vl_setupnn.m'); % Initialize matconvnet13
fprintf('--------------------------------------------------------------------------------\n');
fprintf('--------------------------------------------------------------------------------\n');
% Embedding network: Matching => low distance 
feat_extract_global_triplet_2chstream(1,root_path);
% Similarity network: Matching => high score
feat_extract_global_triplet_2chstream(2,root_path);
