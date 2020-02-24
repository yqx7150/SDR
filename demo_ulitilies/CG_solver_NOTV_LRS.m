function [X,earray1] = CG_solver_NOTV_LRS(b,A, At,w,Lam4,C, X, THRESHOLD,Niter)

oldcost = 0;
earray1 = [];
C.mu1 = 1;
lam1 = 0.5*C.mu1*C.beta1;
%lam1 = 10000;
Lam4=Lam4+1i*0e-18;
eNN=double(0);
for i=1:Niter,
    
    resY = (A(X) - b);
    eY = sum(abs(resY(:)).^2);
    
    resw = X-w; 
    eNN = lam1*sum(abs(resw(:)).^2);
    eNN = eNN + abs(C.mu1*(Lam4(:)'*resw(:))); 
       
    cost1 = eY + eNN; % + eTV;
    
    earray1 = [earray1,cost1];
    
    if(abs(cost1-oldcost)/abs(cost1) < THRESHOLD)
        i
        break;
    end
    oldcost = cost1;
    
  %  conjugate gradient direction
   % ------------------------------  
    % gradient: gn
    
    gn = At(A(X)-b) + lam1*(X-w);  % + lam2*Dt(resTV);
    gn = 2*gn+C.mu1*Lam4;  % + C.mu2*Dt(LamTV);
    
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
    Asn = A(sn);  
    numer = Asn(:)'*resY(:) + lam1*sn(:)'*resw(:) + 0.5* C.mu1*sn(:)'*Lam4(:);    
    denom = Asn(:)'*Asn(:) + lam1*sn(:)'*sn(:); 
    if(denom < 1e-18)
        break;
    end
    alpha = -real(numer)/real(denom);
   
    X = (X + alpha*sn);
end

    
