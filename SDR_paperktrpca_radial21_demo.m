%   Distribution code Version 1.0 -- 06/22/2015 
%%   The current version is not optimized.
%   All rights reserved.
%   This work should only be used for nonprofit purposes.
%
%   Please cite the paper when you use th code:
%   The Code is created based on the method described in the following paper 
%   [1] "Sparse and Dense Hybrid Representation via Subspace Modeling for Dynamic MRI", Qiegen Liu,Jingfei He, Dong Liang

%   Copyright 2015, Nanchang University.
%   The code and the algorithm are for non-comercial use only.
%%   the demo produces the results shown in Fig. 14 of the ref.[1].

clear;close all; 
path(path,'../demo_ulitilies/');

%% load the fully sampled cartesian data
load phan.mat
x=X(:,:,1:80);
IDEAL = x; 
x=double(x);
[n1,n2,n3]=size(x);

m=n1*n2;
n=n3;
% for iiii = 1:n3
%     figure(22);imshow(abs(x(:,:,iiii)),[]);
% end

x= reshape(x,m,n);
%% load the k-t radial sampling pattern 
%mask = strucrand(n1,n2,n3,21);
load mask_21; mask = mask_21;
mask = fftshift(fftshift(mask,1),2);
S=find(mask~=0);
sum(mask(:)~=0)/n1/n2/n3
for iiii = 1:n3
    figure(22);imshow(fftshift(mask(:,:,iiii)),[]);
end

%% A and At operators are the forward and backward Fourier sampling operators respectively
A = @(z)A_fhp3D(z,S,n1,n2,n3);
At=@(z)At_fhp3D(z,S,n1,n2,n3);

%% first guess
b = A(x);
x_init = At(b);
x_init1=reshape(x_init,n1,n2,n3);

%% r denotes the number of temporal basis functions in the dictionary
r = 70;
%% The algorithm parameters are all specified in opts.
opts.outer = 7;% The iterations of the outer loop
opts.lambda1=12e-6/(n1*n2);% The regularization parameter (on the l2 norm of U)
opts.lambda1_2=28e3/(n1*n2);% The regularization parameter (on the l1 norm of U)
opts.lambda2=1e-8/(n1*n2);% The regularization parameter (on the l2 norm of V)
opts.beta1=1e-4; % continuation parameter for the l1 norm; initialize it
part_1 = 3;
opts.inner_iter = 50; % no of inner iterations
opts.betarate = 15; % similar increment for TV norm
opts.toltau = 2e-2;

%% Initialize the U and V matrices; (spatial weights/coefficients and temporal bases)
% V is initialized as a random matrix; and U is initialized from the below
V = double(rand(r,n));

%% This is the CG(conjugate gradient) algorithm to solve the U subproblem
[U,earray_u] = xupdateUal(b,A, At,zeros(m,r), V,0,opts,1e-10,30);

%opts.beta1=1./max(abs(U(:))); % Continuation parameter - initialize it
dddd = sum(abs(U));
[ddY,ddI] = sort(dddd,'descend');
U = U(:,ddI);V = V(ddI,:);
UBCS = reshape(U,n1,n2,70);
figure(202);imshow([abs(UBCS(:,:,1)),abs(UBCS(:,:,2)),abs(UBCS(:,:,3)),abs(UBCS(:,:,4)),abs(UBCS(:,:,5)),abs(UBCS(:,:,6)),abs(UBCS(:,:,7)),abs(UBCS(:,:,8)),abs(UBCS(:,:,9)),abs(UBCS(:,:,10)),abs(UBCS(:,:,11)),abs(UBCS(:,:,12)),abs(UBCS(:,:,13)),abs(UBCS(:,:,14)),abs(UBCS(:,:,15))],[]);   
figure(203);imshow([abs(UBCS(:,:,16)),abs(UBCS(:,:,17)),abs(UBCS(:,:,18)),abs(UBCS(:,:,19)),abs(UBCS(:,:,20)),abs(UBCS(:,:,21)),abs(UBCS(:,:,22)),abs(UBCS(:,:,23)),abs(UBCS(:,:,24)),abs(UBCS(:,:,25)),abs(UBCS(:,:,26)),abs(UBCS(:,:,27)),abs(UBCS(:,:,28)),abs(UBCS(:,:,29)),abs(UBCS(:,:,30))],[]);   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Iterate between the subproblems of shrinkage, CG update of U, CG update of V using continuation
X = reshape(x_init,n1,n2,n3); 
X_est = U*V;
A = @(z)A_fhp3D_3D(z,S); % The forward Fourier sampling operator
At = @(z)At_fhp3D_3D(z,S,n1,n2,n3); % The backward Fourier sampling operator
b = A(reshape(x,n1,n2,n3));  
Lam4 = zeros(n1,n2,n3);% Lam4 is the Lagrange multipliers.
earray=[];earraySER=[];o=0;
for out = single(1: opts.outer)  
    o=o+1
    for in = single(1:opts.inner_iter)
    [X,earray1] = CG_solver_NOTV_LRS(b,A,At,reshape(X_est,n1,n2,n3)-Lam4/opts.beta1,zeros([n1,n2,n3]),opts, X, 1e-7,9);

    U1 = U(:,1:part_1);U2 = U(:,part_1+1:end);
    V1 = V(1:part_1,:);V2 = V(part_1+1:end,:);
    %% U subproblem - CG algorithm
    sig1 = get_operator_norm_Liu(V1,0);
    sig1 = opts.beta1*sig1^2;
    for ii = 1:1
        grad1 = opts.beta1*(X_est-reshape(X+Lam4/opts.beta1,m,n))*V1';  %
        U1 = U1-(opts.lambda1*U1+grad1)/(sig1+opts.lambda1);
        X_est = [U1,U2]*V;  %update the latest value
    end
    sig2 = get_operator_norm_Liu(V2,0);
    sig2 = sig2^2;
    for ii = 1:1
        grad2 = (X_est-reshape(X+Lam4/opts.beta1,m,n))*V2';  %
        U2 = soft(U2-grad2/sig2,opts.lambda1_2/sig2/opts.beta1);
        X_est = [U1,U2]*V;  %update the latest value       
    end
    U = [U1,U2];
    
    [V,earray_v] = xupdateVprobLRS_revised(reshape(X+Lam4/opts.beta1,m,n),V, U,opts,1e-7,5);    
    X_est = U*V;
    %% COST Calculations
    dc = A(reshape(X_est,n1,n2,n3))-b; % data consistency
    reg_U1 = sum(abs(U1(:)).^2); % l2 norm on the spatial weights
    reg_U2 = sum(abs(U2(:))); % l1 norm on the spatial weights
    reg_V = sum(abs(V(:)).^2); % l2 norm on the dictionary
    cost = sum(abs(dc(:)).^2) +opts.lambda1*reg_U1+opts.lambda1_2*reg_U2+opts.lambda2*reg_V;
    earray = [earray,cost]
    
    -20*log10( norm(X(:)-x(:))/norm(x(:)) )
    earraySER=[earraySER,-20*log10( norm(X(:)-x(:))/norm(x(:)) )];
    if in>1
        abs(earray(end) - earray(end-1))/abs(earray(end))
        if abs(earray(end) - earray(end-1))/abs(earray(end)) < opts.toltau
            break;
        end
    end
    
    figure(10);
    subplot(2,3,1); imagesc(abs(reshape(U(:,3),n1,n2))); title('Example spatial weight');colormap(gray)
    subplot(2,3,2); imagesc(abs(reshape(x_init(:,3),n1,n2))); title('Example auxilary variable');colormap(gray)
    subplot(2,3,4); plot(abs(V(3,:))); title('Example temporal basis: 1');
    subplot(2,3,3); imagesc(abs(X(:,:,3))); title('A spatial frame: reconstruction');colormap(gray)%reshape(L(:,1),n1,n2,1))); title('L');
    subplot(2,3,5); imagesc(abs(reshape(x(:,3),n1,n2))); title('Original');colormap(gray)%reshape(L(:,1),n1,n2,1))); title('L');
    pause(0.01);
    Lam4 = Lam4 - 1.618*opts.beta1*(reshape(X_est,n1,n2,n3) - X);
    end
    opts.beta1 = opts.beta1*opts.betarate; % update the continuation parameter
end
figure(1991);plot(earray(2:end));xlabel('Iteration #');ylabel('Cost');
figure(1992);plot(earraySER(2:end));xlabel('Iteration #');ylabel('SER(dB)');

% %%%%%% display the L£«S components, the final reconstruction, and the ground truth MRI series
UU = reshape(U,n1,n2,70);
figure(222);imshow([abs(UU(:,:,1)),abs(UU(:,:,2)),abs(UU(:,:,3)),abs(UU(:,:,4)),abs(UU(:,:,5)),abs(UU(:,:,6)),abs(UU(:,:,7)),abs(UU(:,:,8)),abs(UU(:,:,9)),abs(UU(:,:,10)),abs(UU(:,:,11)),abs(UU(:,:,12)),abs(UU(:,:,13)),abs(UU(:,:,14)),abs(UU(:,:,15))],[]);   
figure(223);imshow([abs(UU(:,:,16)),abs(UU(:,:,17)),abs(UU(:,:,18)),abs(UU(:,:,19)),abs(UU(:,:,20)),abs(UU(:,:,21)),abs(UU(:,:,22)),abs(UU(:,:,23)),abs(UU(:,:,24)),abs(UU(:,:,25)),abs(UU(:,:,26)),abs(UU(:,:,27)),abs(UU(:,:,28)),abs(UU(:,:,29)),abs(UU(:,:,30))],[]);   

XXX = reshape(U1*V1,[n1,n2,n3]);
figure(991);imshow([abs(XXX(:,:,1)),abs(XXX(:,:,2)),abs(XXX(:,:,3)),abs(XXX(:,:,4)),abs(XXX(:,:,5)),abs(XXX(:,:,6)),abs(XXX(:,:,7)),abs(XXX(:,:,8)),abs(XXX(:,:,9)),abs(XXX(:,:,10)),abs(XXX(:,:,11)),abs(XXX(:,:,12)),abs(XXX(:,:,13)),abs(XXX(:,:,14)),abs(XXX(:,:,15))],[]);
XXX = reshape(U2*V2,[n1,n2,n3]);
figure(992);imshow([abs(XXX(:,:,1)),abs(XXX(:,:,2)),abs(XXX(:,:,3)),abs(XXX(:,:,4)),abs(XXX(:,:,5)),abs(XXX(:,:,6)),abs(XXX(:,:,7)),abs(XXX(:,:,8)),abs(XXX(:,:,9)),abs(XXX(:,:,10)),abs(XXX(:,:,11)),abs(XXX(:,:,12)),abs(XXX(:,:,13)),abs(XXX(:,:,14)),abs(XXX(:,:,15))],[]);
XXX = X;
figure(993);imshow([abs(XXX(:,:,1)),abs(XXX(:,:,2)),abs(XXX(:,:,3)),abs(XXX(:,:,4)),abs(XXX(:,:,5)),abs(XXX(:,:,6)),abs(XXX(:,:,7)),abs(XXX(:,:,8)),abs(XXX(:,:,9)),abs(XXX(:,:,10)),abs(XXX(:,:,11)),abs(XXX(:,:,12)),abs(XXX(:,:,13)),abs(XXX(:,:,14)),abs(XXX(:,:,15))],[]);
XXX = IDEAL;
figure(994);imshow([abs(XXX(:,:,1)),abs(XXX(:,:,2)),abs(XXX(:,:,3)),abs(XXX(:,:,4)),abs(XXX(:,:,5)),abs(XXX(:,:,6)),abs(XXX(:,:,7)),abs(XXX(:,:,8)),abs(XXX(:,:,9)),abs(XXX(:,:,10)),abs(XXX(:,:,11)),abs(XXX(:,:,12)),abs(XXX(:,:,13)),abs(XXX(:,:,14)),abs(XXX(:,:,15))],[]);


   

