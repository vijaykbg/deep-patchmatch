function Sim_score = feature_extract_2chstream(patches,net_path)
% pathces: is a is a tensor of size [64x64x2xN] where the patches that 
% needs to be compared are stacked along the 3rd dimension
% Eg: If two patches I1 and I2 are to be compared, then dimension of the
% patches would be [64x64x2](I1 and I2 stacked along the 3rd dimension).
% net_path: network path
% Sim_score : output similarity matrix of size 1xN (high if the patches are matching)
batchsize = 500;
load(net_path);
N = size(patches,4);
datasurr = imresize(patches,[32 32]);
datacent = patches(17:48,17:48,:,:);
datasurr = single(bsxfun(@minus, single(datasurr), averageImage_surr));
datacent = single(bsxfun(@minus, single(datacent), averageImage_cent));
for netind = 1:3  
   netcell{netind} = vl_simplenn_move(netcell{netind}, 'gpu') ;
end
dim = 1;
Sim_score = zeros(N,dim);
tot_batch = ceil(N/batchsize);
for i = 1:tot_batch
    imcent = gpuArray(datacent(:,:,:,(i-1)*batchsize+1:min(i*batchsize,N)));
    imsurr = gpuArray(datasurr(:,:,:,(i-1)*batchsize+1:min(i*batchsize,N)));
    final_res = vl_my_simplenn(netcell,imcent,imsurr);  
    Sim_score((i-1)*batchsize+1:min(i*batchsize,N),:) = gather(squeeze(final_res{3}(end).x));
end
Sim_score = Sim_score';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% vl_my_simplenn_new_test
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function final_res = vl_my_simplenn(netcell, xcent, xsurr, varargin)
opts.res = [];
opts.conserveMemory = false;
opts.sync = false;
opts.disableDropout = false ;
opts.freezeDropout = false ;
opts.accumulate = false;
opts.backPropDepth = +inf ;
opts.cudnn = true;
net = netcell{1};
n = numel(net.layers);
opts = vl_argparse(opts, varargin);
N_net12 = n;

doder = 0;

if opts.cudnn
  cudnn = {'CuDNN'} ;
else
  cudnn = {'NoCuDNN'} ;
end

gpuMode = isa(xcent, 'gpuArray') ;

final_res = cell(1,3);

for netind = 1:3
    switch netind
        case 1
            res = [];
            res(1).x = xcent ;
            net = netcell{netind};            
        case 2
            res = [];
            res(1).x = xsurr;
            net = netcell{netind};
        case 3
            res = [];
            res(1).x = cat(3,final_res{1}(N_net12+1).x,final_res{2}(N_net12+1).x);
            net = netcell{netind};
    end
    n = numel(net.layers) ;
    for i=1:n
      l = net.layers{i} ;
      res(i).time = tic ;
      switch l.type
        case 'conv'
          res(i+1).x = vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, 'pad', l.pad, 'stride', l.stride, cudnn{:}) ;
        case 'pool'
          res(i+1).x = vl_nnpool(res(i).x, l.pool, 'pad', l.pad, 'stride', l.stride, 'method', l.method, cudnn{:}) ;
        case 'relu'
          res(i+1).x = vl_nnrelu(res(i).x) ;       
      end
      % optionally forget intermediate results
      forget = opts.conserveMemory ;
      forget = forget & (~doder || strcmp(l.type, 'relu')) ;
      forget = forget & ~(strcmp(l.type, 'loss') || strcmp(l.type, 'softmaxloss')) ;
      forget = forget & (~isfield(l, 'rememberOutput') || ~l.rememberOutput) ;
      if forget
        res(i).x = [] ;
      end
      if gpuMode & opts.sync
        % This should make things slower, but on MATLAB 2014a it is necessary
        % for any decent performance.
        wait(gpuDevice) ;
      end
      res(i).time = toc(res(i).time) ;
    end
    final_res{netind} = res;
end