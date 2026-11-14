%% Joint torques along the interpolated trajectory (from Problem 6)
% tau = D(q)*qdd + C(q,qd)*qd + G(q), evaluated at every sample of the path.
%
% Expects two .mat files:
%   interpolated_q2.mat : q, qd, qdd (4xM each, rad, rad/s, rad/s^2), tt (1xM, s)
%   dynamics2.mat        : tau (4x1 symbolic torque vector)
% Adjust the file and variable names below if yours differ.

clear; clc;

%% Load trajectory
S   = load('interpolated_q2.mat');
q   = S.q;
qd  = S.qd;
qdd = S.qdd;
tt  = S.tt;
M   = size(q, 2);

%% Load symbolic torque equation and turn it into a fast numeric function
L = load('dynamics2.mat', 'tau');
tau_sym = L.tau;

syms theta_1 theta_2 theta_3 theta_4 real
syms dtheta_1 dtheta_2 dtheta_3 dtheta_4 real
syms ddtheta_1 ddtheta_2 ddtheta_3 ddtheta_4 real
syms g0
theta   = [theta_1,   theta_2,   theta_3,   theta_4];
dtheta  = [dtheta_1,  dtheta_2,  dtheta_3,  dtheta_4];
ddtheta = [ddtheta_1, ddtheta_2, ddtheta_3, ddtheta_4];

% If gravity was left symbolic, plug in its value (mm/s^2 to match g, mm units)
if any(ismember(symvar(tau_sym), g0))
    tau_sym = subs(tau_sym, g0, 9810);
end

tauFun = matlabFunction(tau_sym, 'Vars', {theta, dtheta, ddtheta});

%% Evaluate along the path
unitScale = 1e-9;                      % g*mm^2/s^2 -> N*m (set to 1 if already SI)
tauPath = zeros(4, M);
for k = 1:M
    tauPath(:,k) = unitScale * tauFun(q(:,k)', qd(:,k)', qdd(:,k)');
end

%% Plot
nKnots = 5;                            % 5 for the original Problem 6 knots
knots  = linspace(0, max(tt), nKnots);
names  = {'$\tau_1$', '$\tau_2$', '$\tau_3$', '$\tau_4$'};
 
% Figure sized for the report: 16 cm wide = full text width on A4
fig = figure('Name','Joint torques along interpolated path','NumberTitle','off', ...
             'Units','centimeters', 'Position',[2 2 16 11]);
tl = tiledlayout(fig, 2, 2, 'TileSpacing','compact', 'Padding','compact');
 
% Each torque on its own axis, so small variations stay visible
for i = 1:4
    nexttile; hold on; grid on;
    plot(tt, tauPath(i,:), 'LineWidth', 1.2);
    xline(knots, ':k');
    xlim([0 max(tt)]);
    title(names{i}, 'Interpreter', 'latex');
    ylabel('$\tau$ [N$\cdot$m]', 'Interpreter', 'latex');
    if i > 2
        xlabel('$t$ [s]', 'Interpreter', 'latex');   % x label only on bottom row
    end
end
 
set(findall(fig, '-property', 'FontSize'), 'FontSize', 9);
set(findall(fig, '-property', 'TickLabelInterpreter'), 'TickLabelInterpreter', 'latex');
 
% Export as vector PDF for LaTeX (create the Figures folder first if needed)
% exportgraphics(fig, 'Figures/joint_torques.pdf', 'ContentType', 'vector');
 
%% Summary
fprintf('Peak |tau| per joint [N*m]:\n');
fprintf('  tau_%d: %.4e\n', [1:4; max(abs(tauPath), [], 2)']);