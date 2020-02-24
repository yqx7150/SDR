%--------------------------------------------------------------------------
% CG solution to region limited problem
% [X,Potential,ErrorArray,ErrorIndex,efinal] = xupdateHOTV(A,b,baseline,mask,kappa,lambda,mu,Niter,Potential)
% Solves {X*} = arg min_{X} ||Af-b||^2 + mu ||Rf||_{l_1}
%--------------------------------------------------------------------------

function [X,earray1] = xupdateVprob(b,A, At,X, U,C,THRESHOLD,Niter)

oldcost = 0;
earray1 = [];

lam1 = double(0.5*C.lambda2);
for i=double(1:Niter),
    
    resY = (A(U*X) - b);
    eY = sum(abs(resY(:)).^2);
    
    
    
    resR = X; 
    
    
    
    cost1 = eY + lam1*sum(abs(resR(:)).^2); %+ eNN + eTV;
    
    earray1 = [earray1,cost1];
    
    if(abs(cost1-oldcost)/abs(cost1) < THRESHOLD)
        i;
        break;
    end
    oldcost = cost1;
    
  %  conjugate gradient direction
   % ------------------------------
    
    % gradient: gn
    
    gn = U'*At(resY) + lam1*(X);% + lam2*Dt(resTV);
    gn = 2*gn;%+C.mu1*Lam4 + C.mu2*Dt(LamTV);
    
    % search direction: sn  
    if(i==1)
        sn = gn;                                          
        oldgn = gn;
    else
        gamma = abs(sum(gn(:)'*gn(:))/sum(oldgn(:)'*oldgn(:)));
        sn = gn + gamma*sn; 
        oldgn = gn;
    end
    
    % line search
    %-------------
    Asn = A(U*sn);  
    %Dsn = D(sn);
    
    numer = Asn(:)'*resY(:)+ lam1*sn(:)'*resR(:); %+ 0.5* C.mu1*sn(:)'*Lam4(:);
%     for index = 1:3
%          numer = numer + lam2*sum(sum(sum(conj(resTV{index}).*Dsn{index})))   +  0.5*C.mu2*sum(sum(sum(conj(LamTV{index}).*Dsn{index}))); 
%     end
    
    denom = Asn(:)'*Asn(:) + lam1*sn(:)'*sn(:); 
%     for index = 1:3 
%        denom = denom + lam2*sum(sum(sum(conj(Dsn{index}).*Dsn{index}))); 
%     end
    if(denom < 1e-18)
     %   break;
     end
    alpha = -real(numer)/real(denom);
   
    % updating
    %-------------
    
    X = (X + alpha*sn);
end

    
