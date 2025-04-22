%% Figure 3 for Radar2025
% Figure to illustrate multi-target detection statistics

%% Simulation: over-sampled DFT (rather than, say, Gaussian A matrix)
snrdb = 25;% signal to noise ratio, in dB (per mode)
n = 2^9;
m = floor(n/3); %select 1/3 rows at random (I.e., 3x oversampled grid)
K = 11; %number of reflectors

% Forward operator: random rows of DFT
A = exp(-1j*2*pi*(0:(n-1))'*(0:(n-1))/n);
rows = randi(n,ceil(1.5*m),1);rows=unique(rows,'stable');
rows = sort(rows(1:m),'ascend');
A = A(rows(:),:);
A = A/sqrt(m);%normalize columns

% Reflectors at two RCS levels and off-grid
% Simulate IQ data
ampl_hi = 10^((snrdb+20*log10(m))/20);%IQ noise unit variance; snr per mode
ampl_lo = 10^((snrdb-10+20*log10(m))/20); %10dB lower on half of reflectors
% start with locations on-grid:
T=randi(n,ceil(1.5*K),1);T=unique(T,'stable');
T=T(1:K);
xtrue=zeros(n,1);
xtrue(T)=ampl_lo;%amplitudes, low
xtrue(T(1:floor(K/2))) = ampl_hi;%amplitudes, high
% for off-grid points,perturb columns of A matrix
points = T - 1 + 0.20*rand(size(T));
Aoff = exp(-1j*2*pi*(0:(n-1))'*points'/n)/sqrt(m);
Aoff = Aoff(rows(:),:);
b = Aoff*xtrue(T);
% add noise (CAWGN)
b = b + (1/sqrt(2))*(randn(m,1)+1j*randn(m,1));

%% Algorithm
W = eye(m);%unit variance noise
options.lambda = 0.1;
options.penalty = 'q';
options.q = 0.8;
options.epsilon = 0.001;
options.cgiter = 5;
options.thresh = 1e-5;
options.ncheck = 50;
options.maxiter = 3000;
options.verbose = 0; %zero for false

% Matched Filter
x_mf = A'*b;

% IRLS
tic;
[xhat,costvals] = IRLS(b,zeros(n,1),options,W,A);
Time.IRLS=toc;
disp(Time.IRLS)
disp(options)

% Figures
figure;
plot(1:n,20*log10(abs(x_mf))-20*log10(m),'rx','LineWidth',2);
hold on;grid on;
plot(1:n,20*log10(abs(xtrue))-20*log10(m),'bo','LineWidth',2)
legend('Matched Filter','true','Fontsize',16,'location','northwest')
ylabel('Amplitude (dB)','FontSize',16)
ax=gca;ax.FontSize = 16;
axis([1 n (snrdb-35) (snrdb+5)])

figure;
plot(1:n,20*log10(abs(xhat))-20*log10(m),'rx','LineWidth',2);
hold on;grid on;
plot(1:n,20*log10(abs(xtrue))-20*log10(m),'bo','LineWidth',2)
legend('IRLS','true','Fontsize',16,'location','northwest')
ylabel('Amplitude (dB)','FontSize',16)
ax=gca;ax.FontSize = 16;
axis([1 n (snrdb-35) (snrdb+5)])

figure;
plot((0:length(costvals)-1)*options.ncheck,costvals,'LineWidth',2);grid on;
title('IRLS Cost Function')
ylabel('Cost function','FontSize',16)
xlabel('Iteration number','FontSize',16)
ax=gca;ax.FontSize = 16;
