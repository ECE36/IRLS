function [x,costvals] = IRLS(b,x0,options,W,A,AH)
% Iteratively Reweighted Least-Squares
% 
% Computes argmin_x  0.5*||W*(Ax - b)||_2^2 + lambda*g(x)
% via a quadratic majorization-minimization approach.
% The penalty function g(x) is sum(abs(x)^2 + epsilon)^(q/2), which is 
% convex for q>=1 (local min found for q<1). This implementation supports 
% complex-valued data.
%
% For references and background, please see tutorial in Potter et al., IEEE
% Radar Conference 2025. Please cite the paper when using this code.
%
%   Code inputs:
%   b       column vector of the data
%   x0      initial value
%   W       whitening operator, R^(-1/2), where R is data covariance matrix
%   A, AH   function handles that compute A*x and A'*y
%           Alternatively, the matrix A can be given explicitly, in which
%           case AH is omitted from the function call.
%   options structure for option values
%       options.lambda  = scalar regularization parameter
%       options.q       = value of q to use in l_q quasi-norm penalty
%       options.cgiter  = number of conjugate gradient (inner) iterations
%       options.maxiter = maximum number of allowed (outer) iterations
%       options.epsilon = smoothing of functional at origin
%       options.thresh  = stopping criterion, ratio of cost function change
%                         to cost function value. Default is 1e-6
%       options.ncheck  = how often to test the threshold. By default,
%                         the value is 20. Set options.ncheck = n > 1 to
%                         check every nth iteration. Make the value
%                         larger than maxiter to avoid checks altogether.
%                         (The cost of computing the cost function is a
%                         significant fraction of the cost of an iteration.
%                         Thus, if many iterations are required, checking
%                         less often reduces computation time.)
%
%   Code Outputs:
%   x           estimate of the cost function's argmin.
%   costvals    vector of computed cost function values. 
%               Only updated every options.ncheck iterations.
%
%   Comments:
%       * Flexibility of options could be removed to reduce overhead at
%         execution. Other code optimization possible, as well.

%   LC Potter, potter.36@osu.edu

%% Input checks
% forward operator as function handle, as needed
if( nargin ==  5 )
    Amat = A;
    A = @(z) Amat*z;
    AH = @(z) Amat'*z;
elseif( nargin < 5 )
    error('At least five arguments are required for IRLS.m.')
end

% prewhitening operator as function handle, as needed
if(~isa(W,'function_handle'))
    Wmat=W;
    W = @(z) Wmat*z;
end

% Verify that options are defined and set defaults.
% Check iterations
if ~isfield(options,'maxiter')
    options.maxiter = length(x0);
end
% Check inner iterations
if ~isfield(options,'cgiter')
    options.cgiter = 5;
end
%Check thresh
if ~isfield(options,'thresh')
    options.thresh = 1e-5;
end
%Check ncheck
if ~isfield(options,'ncheck')
    options.ncheck = 20;
else
    options.ncheck = max(floor(options.ncheck),1);
end
%Check epsilon for smoothing
if ~isfield(options,'epsilon')
    options.epsilon = 1e-3;
end

%% Iteration
%Initialize variables
x = x0;n = length(x);% initialization
Wb = W(b);% pre-whiten the data
stop = 0;
iter_count = 0;
%and, initialize for monitoring the cost function
costvals = zeros(floor(options.maxiter/options.ncheck)+1,1);
tmp = sum((abs(x).^2+options.epsilon).^(options.q/2));
ltilde = options.q*options.lambda;
sqrtltilde = sqrt(ltilde);
costold = 0.5*norm(W(A(x)) - Wb)^2 + options.lambda*tmp; 
costvals(1) = costold;
costcount=2;

%Step through iterations
while ~stop    
    %Compute new weights
    wgt = (abs(x).^2 + options.epsilon).^(1-options.q/2);
    sqrtwgt = sqrt(wgt);

    %A few conjugate gradient steps & increment iteration count
    C = @(z) [sqrtwgt.*AH(W(z)) ; sqrtltilde*z];
    CH = @(z1,z2) W(A(sqrtwgt.*z1))+sqrtltilde*z2;    
    %invoke cg solver
    theta = cg4irls(C,CH,Wb,n,sqrtltilde,options.cgiter);
    x = wgt.*AH(W(theta));
    iter_count = iter_count + 1;

    %Check max iterations
    if( iter_count > options.maxiter )
        stop = 1;
    elseif( mod(iter_count,options.ncheck) == 0 )
        tmp = sum((abs(x).^2+options.epsilon).^(options.q/2));
        costnew = 0.5*norm(W(A(x)) - Wb)^2 + options.lambda*tmp;
        costchange = abs( (costnew - costold) / costnew);
        costold=costnew;
        if costchange < options.thresh
            stop = 1;
        end
        costvals(costcount) = costnew;
        costcount = costcount+1;
    end 
end% end while for iterations

%Inform the user of final result
costvals(costcount:end)=[];%trim if terminated
disp(['Total IRLS Iterations completed: ' ...
    num2str(iter_count) ' out of ' num2str(options.maxiter+1) ' permitted'])
end%end of function


function [theta] = cg4irls(C,CH,Wb,n,sig,iter)
% CG4IRLS  compute conjugate gradient steps for inner iterations
theta       = zeros(size(Wb));
s           = [zeros(n,1);-Wb/sig];
res         = -Wb;        % residual
norm2_res   = res'*res;   % norm-squared of residual
direction   = -res;       % direction
numiter = 0;
while(numiter < iter) 
  z             = C(direction);
  alpha         = norm2_res/sum(abs(z).^2); % step size
  theta         = theta + alpha*direction; % update
  s             = s + alpha*z;
  res           = CH(s(1:n),s(n+1:end));
  norm2_res_new = res'*res;
  beta          = norm2_res_new/norm2_res; % error ratio
  norm2_res     = norm2_res_new; % update squared norm of residual
  direction     = -res + beta*direction; % new direction
  numiter       = numiter+1;
end
end%end function