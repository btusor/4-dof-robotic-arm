clc; 
clear; 
close all;

%% Problem 7: Path of the End-Effector
% Define symbolic variables and DH parameters
syms theta1 theta2 theta3 theta4 d1 a2 a3 a4 real

dhTable = [
    theta1, d1, 0, pi/2;
    theta2, 0, a2, 0;
    theta3, 0, a3, 0;
    theta4, 0, a4, 0
];

% Forward kinematics
T0_1 = simplify(dh_transform(dhTable(1,1), dhTable(1,2), dhTable(1,3), dhTable(1,4)));
T0_2 = simplify(T0_1 * dh_transform(dhTable(2,1), dhTable(2,2), dhTable(2,3), dhTable(2,4)));
T0_3 = simplify(T0_2 * dh_transform(dhTable(3,1), dhTable(3,2), dhTable(3,3), dhTable(3,4)));
T0_4 = simplify(T0_3 * dh_transform(dhTable(4,1), dhTable(4,2), dhTable(4,3), dhTable(4,4)));

% Assemble Stylus Jacobian (6x4)
o0 = [0; 0; 0]; z0 = [0; 0; 1];
o1 = T0_1(1:3, 4); z1 = T0_1(1:3, 3);
o2 = T0_2(1:3, 4); z2 = T0_2(1:3, 3);
o3 = T0_3(1:3, 4); z3 = T0_3(1:3, 3);
o4 = T0_4(1:3, 4);

J_Stylus = [
    cross(z0, o4 - o0), cross(z1, o4 - o1), cross(z2, o4 - o2), cross(z3, o4 - o3);
    z0,                 z1,                 z2,                 z3
];
J_Stylus = simplify(combine(expand(J_Stylus), 'sincos'));

% Geometric link lengths (mm) and circle parameters
linkVars = [d1, a2, a3, a4];
linkVals = [50, 93, 93, 50];
xz = 0;
p_center = [150; 0; 120];
R = 32;

phi = linspace(0, 2*pi, 37);
circlePointsFull = p_center + R * [zeros(size(phi)); cos(phi); sin(phi)];
selectedPoints = [phi(1), phi(10), phi(19), phi(28), phi(37)];
targetPoints = p_center + R * [zeros(size(selectedPoints)); cos(selectedPoints); sin(selectedPoints)];

% Compute inverse kinematics and Jacobians at knot points
selectedJointConfigurations = zeros(4, numel(selectedPoints));
JStylusSelected = cell(1, numel(selectedPoints));
for pointIndex = 1:numel(selectedPoints)
    q_val = reverseKinematics(xz, targetPoints(:, pointIndex));
    selectedJointConfigurations(:, pointIndex) = q_val;
    JStylusSelected{pointIndex} = double(subs(J_Stylus, ...
        [theta1, theta2, theta3, theta4, linkVars], ...
        [q_val.', linkVals]));
end

% Convert Cartesian velocities to joint velocities
v0 = [0; 0; 0]; vA = [0; -27; 0]; vB = [0; 0; -27]; vC = [0; 27; 0]; vD = [0; 0; 0];
cartesianVelocities = [v0, vA, vB, vC, vD];
jointVelocities = zeros(4, size(cartesianVelocities, 2));
for pointIndex = 1:size(cartesianVelocities, 2)
    Jv = JStylusSelected{pointIndex}(1:3, :);
    A_mat = [Jv; 0, 1, 1, 1];
    b_vec = [cartesianVelocities(:, pointIndex); 0];
    jointVelocities(:, pointIndex) = A_mat \ b_vec;
end

% Pre-compile forward kinematics into a fast numeric function
fk_num = matlabFunction(subs(T0_4(1:3, 4), linkVars, linkVals), ...
    'Vars', {[theta1, theta2, theta3, theta4]});

t_segments = [0, 2, 4, 6, 8];
numSegments = numel(selectedPoints) - 1;
timeVector = linspace(t_segments(1), t_segments(end), 400);

%% 1. Quintic Interpolation Evaluation & Plot
a = [0; 0; 0; 0];
A_quintic = cell(1, numSegments);
for seg = 1:numSegments
    A_quintic{seg} = trajectoryPlan(0, 2, ...
        selectedJointConfigurations(:, seg), jointVelocities(:, seg), a, ...
        selectedJointConfigurations(:, seg + 1), jointVelocities(:, seg + 1), a);
end

jointTrajectoryQuintic = zeros(4, numel(timeVector));
cartesianTrajectoryQuintic = zeros(3, numel(timeVector));
for pointIndex = 1:numel(timeVector)
    segmentIndex = min(find(timeVector(pointIndex) <= t_segments(2:end), 1), numSegments);
    t = timeVector(pointIndex) - t_segments(segmentIndex);
    q_t = A_quintic{segmentIndex} * [1; t; t^2; t^3; t^4; t^5];
    jointTrajectoryQuintic(:, pointIndex) = q_t;
    cartesianTrajectoryQuintic(:, pointIndex) = fk_num(q_t.');
end

figure('Color', 'w');
plot3(cartesianTrajectoryQuintic(1, :), cartesianTrajectoryQuintic(2, :), cartesianTrajectoryQuintic(3, :), 'r-', 'LineWidth', 2);
hold on;
plot3(circlePointsFull(1, :), circlePointsFull(2, :), circlePointsFull(3, :), 'b--', 'LineWidth', 1.2);
plot3(targetPoints(1, :), targetPoints(2, :), targetPoints(3, :), 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6);
grid on; axis equal; xlim([100, 200]);
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
title('End-Effector Path (Quintic Interpolation)');
ax = gca; ax.Color = 'w'; ax.XColor = 'k'; ax.YColor = 'k'; ax.ZColor = 'k';
ax.GridColor = 'k'; ax.GridAlpha = 0.15; ax.Title.Color = 'k';
lgd = legend('Interpolated Trajectory', 'Reference Circle', 'Target Waypoints', 'Location', 'best');
lgd.Color = 'w'; lgd.TextColor = 'k'; lgd.EdgeColor = 'k';
view(45, 30);

%% 2. Cubic Interpolation Evaluation & Plot (Improvement)
A_cubic = cell(1, numSegments);
for seg = 1:numSegments
    A_cubic{seg} = trajectoryPlanCubic(0, 2, ...
        selectedJointConfigurations(:, seg), jointVelocities(:, seg), ...
        selectedJointConfigurations(:, seg + 1), jointVelocities(:, seg + 1));
    fprintf('\nCubic Coefficients at segment %d:\n', seg);
    disp(A_cubic{seg});
end

jointTrajectoryCubic = zeros(4, numel(timeVector));
cartesianTrajectoryCubic = zeros(3, numel(timeVector));
for pointIndex = 1:numel(timeVector)
    segmentIndex = min(find(timeVector(pointIndex) <= t_segments(2:end), 1), numSegments);
    t = timeVector(pointIndex) - t_segments(segmentIndex);
    q_t = A_cubic{segmentIndex} * [1; t; t^2; t^3];
    jointTrajectoryCubic(:, pointIndex) = q_t;
    cartesianTrajectoryCubic(:, pointIndex) = fk_num(q_t.');
end

figure('Color', 'w');
plot3(cartesianTrajectoryCubic(1, :), cartesianTrajectoryCubic(2, :), cartesianTrajectoryCubic(3, :), 'r-', 'LineWidth', 2);
hold on;
plot3(circlePointsFull(1, :), circlePointsFull(2, :), circlePointsFull(3, :), 'b--', 'LineWidth', 1.2);
plot3(targetPoints(1, :), targetPoints(2, :), targetPoints(3, :), 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6);
grid on; axis equal; xlim([100, 200]);
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
title('Improved End-Effector Path (Cubic Interpolation)');
ax = gca; ax.Color = 'w'; ax.XColor = 'k'; ax.YColor = 'k'; ax.ZColor = 'k';
ax.GridColor = 'k'; ax.GridAlpha = 0.15; ax.Title.Color = 'k';
lgd = legend('Interpolated Trajectory', 'Reference Circle', 'Target Waypoints', 'Location', 'best');
lgd.Color = 'w'; lgd.TextColor = 'k'; lgd.EdgeColor = 'k';
view(45, 30);

%% Helper Functions
function T = dh_transform(theta_i, d_i, a_i, alpha_i)
    rotPhi = [cos(theta_i), -sin(theta_i), 0, 0;
              sin(theta_i),  cos(theta_i), 0, 0;
              0,             0,            1, 0;
              0,             0,            0, 1];
    transD_i = [1, 0, 0, 0;
                0, 1, 0, 0;
                0, 0, 1, d_i;
                0, 0, 0, 1];
    transA_i = [1, 0, 0, a_i;
                0, 1, 0, 0;
                0, 0, 1, 0;
                0, 0, 0, 1];
    rotAlpha = [1, 0,            0,             0;
                0, cos(alpha_i), -sin(alpha_i), 0;
                0, sin(alpha_i),  cos(alpha_i), 0;
                0, 0,            0,             1];
    T = rotPhi * transD_i * transA_i * rotAlpha;
end

function q = reverseKinematics(xz, o04)
    d1 = 50; a2 = 93; a3 = 93; a4 = 50;
    ox = o04(1); oy = o04(2); oz = o04(3);

    theta1_val = atan2(oy, ox);
    theta_ee = asin(xz);

    r_ee = sqrt(ox^2 + oy^2);
    r = r_ee - a4 * cos(theta_ee);
    s = oz - d1 - a4 * sin(theta_ee);

    cos_3 = (r^2 + s^2 - a2^2 - a3^2) / (2 * a2 * a3);
    cos_3 = max(-1, min(1, cos_3));
    sin_3 = -sqrt(1 - cos_3^2);
    theta3_val = atan2(sin_3, cos_3);

    alpha = atan2(s, r);
    beta  = atan2(a3 * sin(theta3_val), a2 + a3 * cos(theta3_val));
    theta2_val = alpha - beta;
    theta4_val = theta_ee - theta2_val - theta3_val;

    q = [theta1_val; theta2_val; theta3_val; theta4_val];
end

function A = trajectoryPlan(tA, tB, qA, vA, aA, qB, vB, aB)
    A = zeros(4, 6);
    coefficients = [
        1, tA, tA^2,   tA^3,    tA^4,    tA^5;
        0,  1, 2*tA, 3*tA^2,  4*tA^3,  5*tA^4;
        0,  0,    2,   6*tA, 12*tA^2, 20*tA^3;
        1, tB, tB^2,   tB^3,    tB^4,    tB^5;
        0,  1, 2*tB, 3*tB^2,  4*tB^3,  5*tB^4;
        0,  0,    2,   6*tB, 12*tB^2, 20*tB^3
    ];
    for i = 1:4
        conditions = [qA(i); vA(i); aA(i); qB(i); vB(i); aB(i)];
        polynomialCoefficients = coefficients \ conditions;
        A(i, :) = polynomialCoefficients.';
    end
end

function A = trajectoryPlanCubic(tA, tB, qA, vA, qB, vB)
    A = zeros(4, 4);
    coefficients = [
        1, tA, tA^2,   tA^3;
        0,  1, 2*tA, 3*tA^2;
        1, tB, tB^2,   tB^3;
        0,  1, 2*tB, 3*tB^2
    ];
    for i = 1:4
        conditions = [qA(i); vA(i); qB(i); vB(i)];
        polynomialCoefficients = coefficients \ conditions;
        A(i, :) = polynomialCoefficients.';
    end
end
