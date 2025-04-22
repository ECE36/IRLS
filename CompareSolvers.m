%% Test script to compare solvers: MF, BASL, IRLS, FASTA
% Script computes figure for presentation slide 13, Radar`25.
clearvars

%% Define problem instance
m=250;
n=1000;
K=10;
snrdb = 20;
% Algorithm
options.lambda = 0.1;mu=options.lambda;
options.epsilon = 1e-4;
options.q = 0.7;
options.cgiter = 5;
options.maxiter = 5;
Extra = 1.50;%empirically observed inflation factor so that FASTA exhibits
    % convergence similar to IRLS: steps = Extra*maxiter*(cgiter+1).
%other
options.thresh = 1e-7;
options.ncheck = 10^6;%don't need to save values of objective functions

%% Create data realization
% clutter and noise
Rc = randn(m,m)+1j*randn(m,m);Rc = Rc*Rc';
Rc = Rc/mean(diag(Rc))/0.75;%0.75 variance, on average
Rnoise = 0.25*eye(m);
Wc=sqrtm(Rc)\eye(m);%whitening, clutter
Wnoise = sqrtm(Rnoise)\eye(m);%whitening, noise only
W = sqrtm(Rc+Rnoise)\eye(m);%whitening, combined (clutter and noise indep)
% subsampled DFT operator
A = exp(-1j*2*pi*(0:(n-1))'*(0:(n-1))/n);%Fwd=rows of DFT
rows = randi(n,ceil(1.5*m),1);rows=unique(rows,'stable');
rows = sort(rows(1:m),'ascend');
A = A(rows(:),:);
A = A/sqrt(m);%normalize columns
% reflectors at two RCS levels (here, on grid)
ampl_hi = sqrt(m)*10^((snrdb)/20);%IQ noise unit variance; snr per mode
ampl_lo = sqrt(m)*10^((snrdb-10)/20); %10dB lower on half of reflectors
T=randi(n,ceil(1.5*K),1);T=unique(T,'stable');
T=T(1:K);
xtrue=zeros(n,1);
xtrue(T)=ampl_lo;%amplitudes, lo
xtrue(T(1:floor(K/2))) = ampl_hi;%amplitudes, high
% data plus clutter & noise
b = A*xtrue;
b = b + sqrtm(Rc+Rnoise)*(randn(size(b)) + 1j*randn(size(b)))/sqrt(2);

%% invoke solvers
% initialize with whitened matched filter
x0 = A'*(W*W)*b;

%IRLS
tic
[xIRLS,costvalsIrls] = IRLS(b,x0,options,W,A);
IRLStime = toc;

% BaSL
tic
[xBASL,costvalsBasl] = BASL(b,options,W,Wnoise,Rnoise,A);
BASLtime = toc;

% FASTA using the 'opts' struct
opts = [];
opts.maxIters = round(Extra*options.maxiter*(options.cgiter+1));
opts.tol = 1e-7;  % Use strict tolerance so iterations execute
opts.recordObjective = false; % Record the objective function for plotting
opts.verbose=false;
opts.stringHeader='    '; % Append a tab to all text output from FISTA.  
                          % This option makes formatting look a bit nicer. 
tic
[xFASTA, outs_adapt] = fasta_q(W*A,(W*A)',W*b,options.q,mu*50,x0,opts);
FASTAtime=toc;

%% plots
text=sprintf('Run times:\n   BASL: %g\n    IRLS: %g\n    FASTA: %g',...
    BASLtime,IRLStime,FASTAtime);
disp(text)
% to plot in dB scale, subtract processing gain, 10*log10(m), thus vertical
% axis is 20*log10(A/sigma)
figure;
% Matched filter (no whitening)
subplot(221)
plot(1:n,20*log10(abs(A'*b))-10*log10(m),'rx','LineWidth',2);
hold on;grid on;
plot(1:n,20*log10(abs(xtrue))-10*log10(m),'bo','LineWidth',2)
legend('Matched Filter','true','Fontsize',16,'location','southwest')
ylabel('Amplitude (dB)','FontSize',16)
ax=gca;ax.FontSize = 16;
axis([1 n (snrdb-35) (snrdb+5)])
% IRLS
subplot(222)
plot(1:n,20*log10(abs(xIRLS))-10*log10(m),'rx','LineWidth',2);
hold on;grid on;
plot(1:n,20*log10(abs(xtrue))-10*log10(m),'bo','LineWidth',2)
legend('IRLS','true','Fontsize',16,'location','southwest')
ylabel('Amplitude (dB)','FontSize',16)
ax=gca;ax.FontSize = 16;
axis([1 n (snrdb-35) (snrdb+5)])
% BASL
subplot(223)
plot(1:n,20*log10(abs(xBASL))-10*log10(m),'rx','LineWidth',2);
hold on;grid on;
plot(1:n,20*log10(abs(xtrue))-10*log10(m),'bo','LineWidth',2)
legend('BASL','true','Fontsize',16,'location','southwest')
ylabel('Amplitude (dB)','FontSize',16)
ax=gca;ax.FontSize = 16;
axis([1 n (snrdb-35) (snrdb+5)])
% FASTA
subplot(224)
plot(1:n,20*log10(abs(xFASTA))-10*log10(m),'rx','LineWidth',2);
hold on;grid on;
plot(1:n,20*log10(abs(xtrue))-10*log10(m),'bo','LineWidth',2)
legend('FASTA','true','Fontsize',16,'location','southwest')
ylabel('Amplitude (dB)','FontSize',16)
ax=gca;ax.FontSize = 16;
axis([1 n (snrdb-35) (snrdb+5)])
pos = get(gcf, 'Position');
set(gcf, 'Position',pos+400*[-1, -1, 1, 1])