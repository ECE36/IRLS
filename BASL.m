function [x,costvals] = BASL(b,options,W,Wnoise,Rn,A,AH)
% Background Supplemental Loading (BaSL) iteration with MVDR form.
% Reference: Jones, et al. 2020 IEEE Radar Conference.
%
% For additional references and background, please see tutorial in Potter 
% et al., IEEE Radar Conference 2025. Please cite if using this code.
%
%   Code inputs:
%   b       column vector of the data
%   W       whitening filter for clutter plus noise
%   Wnoise  whitening filter for noise covariance only
%   A, AH   function handles that compute A*x and A'*y
%           Alternatively, the matrix A can be given explicitly, in which
%           case AH is omitted from the function call.
%   options structure for option values
%       options.cgiter  = number of conjugate gradient (inner) iterations
%       options.maxiter = maximum number of allowed (outer) iterations
%       options.q       = in (0,1) for monitoring related cost function
%       options.epsilon = smoothing; for monitoring a related cost function
%       options.lambda  = Lagrange multiplier for related cost function
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

%   LC Potter, potter.36@osu.edu

%% Input checks
% forward operator as function handle, as needed
if( nargin ==  6 )
    Amat = A;
    A = @(z) Amat*z;
    AH = @(z) Amat'*z;
elseif( nargin < 6 )
    error('At least five arguments are required for BASL.m.')
end

% prewhitening operator as function handle, as needed
if(~isa(W,'function_handle'))
    Wmat=W;
    W = @(z) Wmat*z;
end
if(~isa(Wnoise,'function_handle'))
    Wnoisemat=Wnoise;
    Wnoise = @(z) Wnoisemat*z;
end

% Verify that options are defined and set defaults.
% Check iterations
if ~isfield(options,'maxiter')
    options.maxiter = 5;%default from Jones paper
end
% Check inner iterations
if ~isfield(options,'cgiter')
    options.cgiter = 5;
end
%Check ncheck
if ~isfield(options,'ncheck')
    options.ncheck = 20;
else
    options.ncheck = max(floor(options.ncheck),1);
end

%% Iteration
%Initialize variables
x = AH(b);% initialization with white-noise matched filter (per UK papers)
n = length(x);
Wb = W(b);% prewhiten data using clutter plus noise, to save repetition
iter_count = 0;
%and, initialize for monitoring a related cost function
costvals = zeros(floor(options.maxiter/options.ncheck)+1,1);
tmp = sum((abs(x).^2+options.epsilon).^(options.q/2));
costold = 0.5*norm(W(A(x)) - Wb)^2 + options.lambda*tmp; 
costvals(1) = costold;
costcount=2;

%Step through iterations
while (iter_count < options.maxiter)    
    %Compute new weights
    wgt = abs(x).^2;
    sqrtwgt = sqrt(wgt);

    %Numerator: a few conjugate gradient steps
    C = @(z) [sqrtwgt.*AH(W(z)) ; z];
    CH = @(z1,z2) W(A(sqrtwgt.*z1)) + z2;    
    %invoke cg solver
    theta = cg4basl(C,CH,Wb,n,options.cgiter);
    x = AH(W(theta));
    % need BaSL's MVDR-inspired scaling -- the denominator term
    % Possible that warm-start CG might reduce computation load over "\"
    tmp = (Amat*diag(wgt)*Amat' + Rn)\Amat;
    x = x./diag(Amat'*tmp);
    iter_count = iter_count + 1;
    
    if( mod(iter_count,options.ncheck) == 0 )
        tmp = sum((abs(x).^2+options.epsilon).^(options.q/2));
        costnew = 0.5*norm(W(A(x)) - Wb)^2 + options.lambda*tmp;
        costvals(costcount) = costnew;
        costcount = costcount+1;
    end 
end% end while for iterations
costvals(costcount:end)=[];%trim if terminated
end%end of function


function [theta] = cg4basl(C,CH,Wb,n,iter)
% CG4BASL  compute conjugate gradient steps for inner iterations
theta       = zeros(size(Wb));
s           = [zeros(n,1);-Wb];
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