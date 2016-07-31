function feat_extract_global_triplet_2chstream(method,root_path)
TrainSetName = 'liberty'; % Alternatively models learned on yosemite or notredame can be used here
Data_dir = fullfile(root_path,'data/data.mat');
Net_dir_embedding = fullfile(root_path,'models/embedding/model_global_triplet_%s.mat');
Net_dir_2chstream = fullfile(root_path,'models/2chstream/model_2chstream_%s.mat');

load(Data_dir);
net_path_embedding = sprintf(Net_dir_embedding,TrainSetName);
net_path_2chstream = sprintf(Net_dir_2chstream,TrainSetName);

    
switch method
    case 1
        Desc = feature_extract(data,net_path_embedding); % Compute descriptors
        % squared L2 distance between descriptors
        PatchDist = sum((Desc(:, PatchesIdx1) - Desc(:, PatchesIdx2)) .^ 2, 1);
        fprintf(' Distances || gt labels \n');
        disp([PatchDist' labels]);        
    case 2
        N_labels = length(PatchesIdx1);
        ip_patches = zeros(64,64,2,N_labels,'uint8');
        ip_patches(:,:,1,:) = data(:,:,PatchesIdx1);
        ip_patches(:,:,2,:) = data(:,:,PatchesIdx2);
        % Similarity score
        PatchSim = feature_extract_2chstream(ip_patches,net_path_2chstream);
        fprintf('Similarity || gt labels \n');
        disp([PatchSim' labels]);
end
fprintf('--------------------------------------------------------------------------------\n');
fprintf('--------------------------------------------------------------------------------\n');
