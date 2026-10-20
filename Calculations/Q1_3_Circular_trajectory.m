clear all;
close all;
clc;

%% Robot parameters [mm]

d_1 = 50;
a_2 = 93;
a_3 = 93;
a_4 = 50;

%% Circular trajectory inverse kinematics

p_c = [150; 0; 120];   % circle centre in frame {0} [mm]
R   = 32;              % circle radius [mm]
x_z0 = 0;              % stylus horizontal: last component of x_4^0 = 0

N   = 37;                          % j = 0,1,...,36
phi = linspace(0, 2*pi, N);        

o_40 = zeros(3, N);                % target tip positions [mm]
q    = zeros(4, N);                % joint configurations [rad]

for j = 1:N
    o_40(:, j) = p_c + R*[0; cos(phi(j)); sin(phi(j))];
    q(:, j)    = inverseKinematics(o_40(:, j), x_z0, d_1, a_2, a_3, a_4);
end

%% Check with forward kinematics

pos_err = zeros(1, N);
x_z     = zeros(1, N);

for j = 1:N
    [o_fk, x_fk] = forwardKinematics(q(:, j), d_1, a_2, a_3, a_4);
    pos_err(j) = norm(o_fk - o_40(:, j));
    x_z(j)     = x_fk(3);
end

fprintf('Max position error: %.2e mm\n', max(pos_err));
fprintf('Max |x_z|:          %.2e\n', max(abs(x_z)));

%% Table of the 37 configurations [deg]

T = array2table([(0:N-1)', rad2deg(phi'), rad2deg(q')], ...
    'VariableNames', {'j', 'phi', 'q1', 'q2', 'q3', 'q4'});
disp(T);

%% Plots

set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');

figure; hold on; grid on; axis equal; view(135, 20);
plot3(o_40(1,:), o_40(2,:), o_40(3,:), 'r.-', 'MarkerSize', 12);

% Draw every 4th configuration
for j = 1:4:N   
    P = jointPositions(q(:, j), d_1, a_2, a_3, a_4);
    plot3(P(1,:), P(2,:), P(3,:), 'k-o', 'LineWidth', 1.2, 'MarkerSize', 4);
end

xlabel('$x_0$ [mm]'); ylabel('$y_0$ [mm]'); zlabel('$z_0$ [mm]');
title('Stylus tip tracking the circle');

figure; plot(0:N-1, rad2deg(q'), '.-', 'LineWidth', 1.2); grid on;
xlabel('$j$'); ylabel('Joint angle [$^\circ$]');
legend('$q_1$', '$q_2$', '$q_3$', '$q_4$', 'Location', 'best');
title('Joint configurations along the circle');

%% Local functions

function q = inverseKinematics(o_40, x_z0, d_1, a_2, a_3, a_4, elbow)

%   o_40  : desired stylus tip position [o_x; o_y; o_z] in frame {0}
%   x_z0  : desired last component of x_4^0, -1 <= x_z0 <= 1
%   elbow : 'up' (default) or 'down'
%   q     : joint angles [theta1; theta2; theta3; theta4] in radians

    if nargin < 7
        elbow = 'up';
    end
    o_x = o_40(1);  o_y = o_40(2);  o_z = o_40(3);

    theta1   = atan2(o_y, o_x);                     % Step 1: base angle
    theta_ee = asin(x_z0);                          % Step 2: stylus angle

    r_ee = sqrt(o_x^2 + o_y^2);                     % Step 3: wrist centre
    r = r_ee - a_4*cos(theta_ee);
    s = o_z - d_1 - a_4*sin(theta_ee);

    c3 = (r^2 + s^2 - a_2^2 - a_3^2) / (2*a_2*a_3); % Step 4: law of cosines
    if abs(c3) > 1
        error('inverseKinematics:unreachable', 'Target point is out of reach.');
    end
    if strcmpi(elbow, 'up')
        s3 = -sqrt(1 - c3^2);
    else
        s3 =  sqrt(1 - c3^2);
    end
    theta3 = atan2(s3, c3);

    theta2 = atan2(s, r) - atan2(a_3*s3, a_2 + a_3*c3);  
    theta4 = theta_ee - theta2 - theta3;                 

    q = [theta1; theta2; theta3; theta4];
end

function [o_40, x_40] = forwardKinematics(q, d_1, a_2, a_3, a_4)
    t1 = q(1);  t2 = q(2);  t23 = q(2)+q(3);  t234 = q(2)+q(3)+q(4);
    rho  = a_2*cos(t2) + a_3*cos(t23) + a_4*cos(t234);
    o_40 = [rho*cos(t1); rho*sin(t1); d_1 + a_2*sin(t2) + a_3*sin(t23) + a_4*sin(t234)];
    x_40 = [cos(t1)*cos(t234); sin(t1)*cos(t234); sin(t234)];
end

function P = jointPositions(q, d_1, a_2, a_3, a_4)
    t1 = q(1);  t2 = q(2);  t23 = q(2)+q(3);  t234 = q(2)+q(3)+q(4);
    u = [cos(t1); sin(t1)];              
    rho = cumsum([0, 0, a_2*cos(t2), a_3*cos(t23), a_4*cos(t234)]);
    h   = cumsum([0, d_1, a_2*sin(t2), a_3*sin(t23), a_4*sin(t234)]);
    P = [u*rho; h];
end
