%% Figure 1 for Radar2025
% Figure to display various scalar penalty functions:
%       q=1 norm
%       q=0.6 quasi-norm
%       log(abs(x)^2)
%       log(abs(x))
%       zero norm
% Scale and shift scalar functions for common display.
clear vars
figure
q=0.6;epsilon=0.001;
x = -2.5:0.001:2.5;[~,indx]=find(x==1);
%
plot(x,abs(x),'LineWidth',2);%L1
hold on;
%
y2=(abs(x).^2 + epsilon).^(q/2);
plot(x,y2,'LineWidth',2)%q=0.8
%
y4 = log(x.^2+epsilon); y4=y4-min(y4);y4=y4/y4(indx);
plot(x,y4,'LineWidth',2)%Log-squared
%
y3 = log(abs(x)+epsilon); y3=y3-min(y3);y3=y3/y3(indx);
plot(x,y3,'LineWidth',2)%Log
%
plot([-2.5,-0.05],[1,1],'k','LineWidth',2)
plot([0.05,2.5],[1,1],'k','LineWidth',2)
plot(0,1,'ko','MarkerSize',8,'LineWidth',2)
grid on;
axis([-2.5,2.5,0,2])
title('Separable Costs, \phi(x), for \epsilon=0.001','Fontsize',18)
%
legend('L1: abs(x)','L_q, abs(x)^{0.6}','ln(x^2)','ln(x)',...
    'L\_0','','','Fontsize',16,'Location','southwest')
