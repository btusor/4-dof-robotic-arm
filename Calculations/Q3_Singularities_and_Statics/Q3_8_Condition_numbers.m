clc; 
clear; 
close all;

%% Problem 8: Condition Number of the Jacobian
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

% Pre-compile Jacobian into a fast numeric function
J_Stylus_num = matlabFunction(subs(J_Stylus, linkVars, linkVals), ...
    'Vars', {[theta1, theta2, theta3, theta4]});

% 1. Ideal circular path configurations (Problem 3)
phi = linspace(0, 2*pi, 37);
circlePoints = p_center + R * [zeros(size(phi)); cos(phi); sin(phi)];
jointConfigurations = zeros(4, numel(phi));
for pointIndex = 1:numel(phi)
    jointConfigurations(:, pointIndex) = reverseKinematics(xz, circlePoints(:, pointIndex));
end

% Compute condition number along the ideal circular path
cond_ideal = zeros(1, numel(phi));
for i = 1:numel(phi)
    J_eval = J_Stylus_num(jointConfigurations(:, i).');
    cond_ideal(i) = cond(J_eval);
end

% 2. Interpolated trajectory configurations (Problem 7 cubic trajectory)
selectedPoints = [phi(1), phi(10), phi(19), phi(28), phi(37)];
targetPoints = p_center + R * [zeros(size(selectedPoints)); cos(selectedPoints); sin(selectedPoints)];
selectedJointConfigurations = zeros(4, numel(selectedPoints));
for pointIndex = 1:numel(selectedPoints)
    selectedJointConfigurations(:, pointIndex) = reverseKinematics(xz, targetPoints(:, pointIndex));
end

v0 = [0; 0; 0]; vA = [0; -27; 0]; vB = [0; 0; -27]; vC = [0; 27; 0]; vD = [0; 0; 0];
cartesianVelocities = [v0, vA, vB, vC, vD];
jointVelocities = zeros(4, size(cartesianVelocities, 2));
for pointIndex = 1:size(cartesianVelocities, 2)
    J_knot = J_Stylus_num(selectedJointConfigurations(:, pointIndex).');
    Jv = J_knot(1:3, :);
    A_mat = [Jv; 0, 1, 1, 1];
    b_vec = [cartesianVelocities(:, pointIndex); 0];
    jointVelocities(:, pointIndex) = A_mat \ b_vec;
end

t_segments = [0, 2, 4, 6, 8];
numSegments = numel(selectedPoints) - 1;
A_all_segments = cell(1, numSegments);
for seg = 1:numSegments
    A_all_segments{seg} = trajectoryPlanCubic(0, 2, ...
        selectedJointConfigurations(:, seg), jointVelocities(:, seg), ...
        selectedJointConfigurations(:, seg + 1), jointVelocities(:, seg + 1));
end

timeVector = linspace(t_segments(1), t_segments(end), 400);
jointTrajectory = zeros(4, numel(timeVector));
cond_actual = zeros(1, numel(timeVector));

for pointIndex = 1:numel(timeVector)
    segmentIndex = min(find(timeVector(pointIndex) <= t_segments(2:end), 1), numSegments);
    t = timeVector(pointIndex) - t_segments(segmentIndex);
    q_t = A_all_segments{segmentIndex} * [1; t; t^2; t^3];
    jointTrajectory(:, pointIndex) = q_t;
    cond_actual(pointIndex) = cond(J_Stylus_num(q_t.'));
end

% Plot Condition Numbers
figure('Color', 'w');
subplot(2,1,1);
plot(phi, cond_ideal, 'b-', 'LineWidth', 1.5);
grid on;
xlabel('\phi (rad)');
ylabel('Condition Number');
title('Jacobian Condition Number - Ideal Circular Path');

subplot(2,1,2);
plot(timeVector, cond_actual, 'r-', 'LineWidth', 1.5);
grid on;
xlabel('Time (s)');
ylabel('Condition Number');
title('Jacobian Condition Number - Interpolated Cubic Path');

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
