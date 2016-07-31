function Desc = feature_extract(patches,net_path)
% patches: A tensor with size [64x64xN]
% net_path: network path
% Desc : output descriptor matrix of size "dim x N" where dim is the
% dimension of the output descriptors
batchsize = 500;
load(net_path);
N = size(patches,3);
data = single(bsxfun(@minus, single(patches), averageImage));
clear patches
data = permute(data,[1 2 4 3]);
net = vl_simplenn_move(net,'gpu');
dim = length(net.layers{end-1}.weights{2});
Desc = zeros(N,dim);
tot_batch = ceil(N/batchsize);
for i = 1:tot_batch
    im = gpuArray(data(:,:,1,(i-1)*batchsize+1:min(i*batchsize,N)));
    res = vl_simplenn(net,im,[]);  
    Desc((i-1)*batchsize+1:min(i*batchsize,N),:) = gather(squeeze(res(end).x))';
end
Desc = Desc';