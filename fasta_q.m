%   Solve the L1 penalized least-squares problem
%           min  mu*g(x)+.5||Ax-b||^2
%   using the solver FASTA 
%
%   Reference: 
%       T. Goldstein, C. Studer, and R. Baraniuk, A field guide to 
%       forward-backward splitting with a FASTA implementation.  
%       ArXiv:1411.3406v6, 28 Dec 2016. 
%   Adapted from https://github.com/tomgoldstein/fasta-matlab to 
%   incorporate g(x) equal ell-q norm, q <1.  To form the proximal 
%   operators, code is modified from:
%       S. Zhou, X. Xiu, Y. Wang, and D. Peng, Revisiting $L_q$ 
%       $(0 \leq q < 1)$ Norm Regularized Optimization, 
%       ArXiv:2306.14394, 2023.
%   For more details, see Radar25 conference paper, Potter et al., 2025.
% 
%  Inputs:
%    A   : A matrix or function handle
%    At  : The adjoint/transpose of A
%    b   : A column vector of measurements
%    mu  : Scalar regularization parameter
%    x0  : Initial guess of solution, often just a vector of zeros
%    opts: Optional inputs to FASTA
%
%   For this code to run, the solver "fasta.m" must be in your path.


function [ solution, outs ] = fasta_q(A,At,b,q,mu,x0,opts)

%%  Check whether we have function handles or matrices
if ~isnumeric(A)
    assert(~isnumeric(At),'If A is a function handle, then At must be a handle as well.')
end
%  If we have matrices, create handles just to keep things uniform below
if isnumeric(A)
    At = @(x)A'*x;
    A = @(x) A*x;
end

%  Check for 'opts'  struct
if ~exist('opts','var') % if user didn't pass this arg, then create it
    opts = [];
end


%%  Define ingredients for FASTA
%  Note: fasta solves min f(Ax)+g(x).
%  f(z) = .5 ||z - b||^2
f    = @(z) .5*norm(z-b,'fro')^2;
grad = @(z) z-b;
% g(z) = mu*|z|
g = @(x) norm(x,q)*mu;
%proxg(z,t) = argmin t*mu*|x|+.5||x-z||^2
prox = @(x,t) qshrink(x,t*mu,q);

%% Call solver
[solution, outs] = fasta(A,At,f,grad,g,prox,x0,opts);

end


%%  The proximal operator for arbitrary q \in [0,1]
% Solving problem   
%       xopt = argmin_x 0.5*||x-a||^2 + lam*||x||_q^q
% This generalizes the simple shrinkage operator obtained for q=1.
function [ out ] = qshrink( x,tau,q )


% Quantize q for switch construct; some special cases are closed-form.
if( (q < 1.01) && (q > 0.99)), q=1;
elseif((q < 0.01) ), q=0;
elseif((q < 0.51) && (q > 0.49)), q=1/2;
elseif((q < 0.68) && (q > 0.65)), q=2/3;
end

% Prox function is simple in special cases
switch q
    case 1% Tom G.
        out = sign(x).*max(abs(x) - tau,0);   

    case 0 % S. Zhou, PSNP
         t      = sqrt(2*tau);
         T      = find(abs(a)>t);  
         out    = zeros(size(x));
         out(T) = x(T); 

    case 1/2 % S. Zhou, PSNP
         t      = (3/2)*tau^(2/3);
         T      = find(abs(x) > t);
         xT     = x(T);
         phi    = acos( (tau/4)*(3./abs(xT)).^(3/2) );
         px     = (4/3)*xT.*( cos( (pi-phi)/3) ).^2;
         out    = zeros(size(x));
         out(T) = px;

    case 2/3 % S. Zhou, PSNP
         t     = 2*(2*tau/3)^(3/4); 
         T     = find( abs(x) >  t );  
         xT    = x(T);       
         tmp1  = xT.^2/2; 
         tmp2  = sqrt( tmp1.^2 - (8*tau/9)^3 );  
         phi   = (tmp1+tmp2).^(1/3)+(tmp1-tmp2).^(1/3);
         px    = sign(xT)/8.*(sqrt(phi)+sqrt(2*abs(xT)./sqrt(phi)-phi)).^3;
         out    = zeros(size(x));
         out(T) = px;
    otherwise
         [px,T] = NewtonLq(x,tau,q);
         out    = zeros(size(x));
         out(T) = px;    
end

end

function [w,T] = NewtonLq(a,alam,q) % S. Zhou, PSNP

    thresh = (2-q)*alam^(1/(2-q))*(2*(1-q))^((1-q)/(q-2));
    T      = find(abs(a)>thresh); 

    if ~isempty(T)
        zT     = a(T);
        w      = zT;
        maxit  = 1e2;
        q1     = q-1;
        q2     = q-2;
        lamq   = alam*q;
        lamq1  = lamq*q1;

        gradLq = @(u,v)(u - zT + lamq*sign(u).*v.^q1);
        hessLq = @(v)(1+lamq1*v.^q2);
        func   = @(u,v)(norm(u-zT)^2/2+alam*sum(v.^q));

        absw   = abs(w);
        fx0    = func(w,absw); 

        for iter  = 1:maxit
            g     = gradLq(w,absw);
            d     = -g./hessLq(absw); 
            alpha = 1;  
            w0    = w;
            for i    = 1:10
                w    =  w0 + alpha*d;  
                absw = abs(w);
                fx   = func(w,absw);
                if  fx < fx0 - 1e-4*norm(w-w0)^2
                   break; 
                end 
                alpha   = alpha*0.5;
            end
            if  norm(g) < 1e-8; break; end
        end
    else
        w = [];
    end
end