%% Figure 2 for Radar2025
% Figure to illustrate quadratic majorization
%       phi=(abs(x).^2 + epsilon).^(q/2)
figure(2)
q=0.8;
epsilon=0.001;
x0=2;
x = -4:0.01:4;
% the scalar function:
phi=(abs(x).^2 + epsilon).^(q/2);
% quadratic majorization
w0 = q/2*(x0^2+epsilon)^(q/2-1);
c0 = (x0^2+epsilon)^(q/2) - w0*x0^2;
phiMaj = c0+w0*x.^2;
% make plot
plot(x,phi,'k','LineWidth',2);
hold on
plot(x,phiMaj,'r--','LineWidth',2)
% overlay point of tangency
y0 = (abs(x0).^2 + epsilon).^(q/2);
plot(2,y0,'ko','MarkerSize',12,'LineWidth',2)
plot([2,2],[0,y0],'k-.')
title('Quadratic Majorization of \phi(x) at x=2, \epsilon=0.001','FontSize',18)
ax=gca;ax.FontSize = 16;
axis([-4 4 0 4])
grid on