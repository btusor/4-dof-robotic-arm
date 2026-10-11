clear all;
close all;
clc;

syms th1 th2 th3 th4 real

%% Robot geometry Parameters

l_1 = 50; %[mm]
l_2 = 93; %[mm]
l_3 = 93; %[mm]
l_4 = 50; %[mm]
l_5 = 45; %[mm]
l_6 = 35; %[mm]

%% DH parameter matrix [theta, d, a, alpha]

DH_sym = [
    th1, l_1, 0,   pi/2;
    th2, 0,   l_2, 0;
    th3, 0,   l_3, 0;
    th4, 0,   l_4, 0
    ];

% General Denavit-Hartenberg transformation matrix
dh_trans = @(theta, d, a, alpha) [
    cos(theta), -sin(theta)*cos(alpha),  sin(theta)*sin(alpha), a*cos(theta);
    sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta);
    0,        sin(alpha),              cos(alpha),             d;
    0,        0,                    0,                   1
];

% Individual transformation matrices
T10 = dh_trans(DH_sym(1,1), DH_sym(1,2), DH_sym(1,3), DH_sym(1,4));
T21 = dh_trans(DH_sym(2,1), DH_sym(2,2), DH_sym(2,3), DH_sym(2,4));
T32 = dh_trans(DH_sym(3,1), DH_sym(3,2), DH_sym(3,3), DH_sym(3,4));
T43 = dh_trans(DH_sym(4,1), DH_sym(4,2), DH_sym(4,3), DH_sym(4,4));

% Robot stylus cumulative forward kinematic model and simplify
T_total_stylus = simplify(T10 * T21 * T32 * T43);

% Transformation matrix between frame 4 and 5
T54 = [
    0,  0,  0, -15;
    0,  1,  0,  45;
    0,  0,  1,  0;
    0,  0,  0,  1
    ];

% Robot camera cumulative forward kinematic model and simplify
T_total_camera = simplify(T_total_stylus * T54);

%% Inverse Kinematics 

function q = inverseKinematics(o_40, x_z0, d_1, a_2, a_3, a_4, elbow)
%INVERSEKINEMATICS Inverse kinematics of the 4-DOF robot arm (kinematic decoupling).
%   q = inverseKinematics(o_40, x_z0, d_1, a_2, a_3, a_4)
%   q = inverseKinematics(o_40, x_z0, d_1, a_2, a_3, a_4, elbow)
%
%   o_40  : desired stylus tip position [o_x; o_y; o_z] in frame {0}
%   x_z0  : desired last component of x_4^0 (sin of the stylus angle to the
%           horizontal plane), -1 <= x_z0 <= 1
%   d_1, a_2, a_3, a_4 : DH parameters (same length unit as o_40)
%   elbow : 'up' (default) or 'down'
%   q     : joint angles [theta1; theta2; theta3; theta4] in radians
%
%   Assumes the stylus points away from the base (cos(theta_ee) >= 0) and
%   the "front" solution for theta1.
 
    if nargin < 7
        elbow = 'up';
    end
 
    o_x = o_40(1);
    o_y = o_40(2);
    o_z = o_40(3);
 
    % Step 1: base angle
    theta1 = atan2(o_y, o_x);
 
    % Step 2: stylus angle to the horizontal, theta_ee = theta2+theta3+theta4
    theta_ee = asin(x_z0);
 
    % Step 3: wrist centre (joint 4) in the arm plane, relative to the shoulder
    r_ee = sqrt(o_x^2 + o_y^2);
    r = r_ee - a_4*cos(theta_ee);
    s = o_z - d_1 - a_4*sin(theta_ee);
 
    % Step 4: elbow angle (law of cosines)
    c3 = (r^2 + s^2 - a_2^2 - a_3^2) / (2*a_2*a_3);
    if abs(c3) > 1
        error('inverseKinematics:unreachable', 'Target point is out of reach.');
    end
    if strcmpi(elbow, 'up')
        s3 = -sqrt(1 - c3^2);   % elbow-up:   theta3 < 0
    else
        s3 =  sqrt(1 - c3^2);   % elbow-down: theta3 > 0
    end
    theta3 = atan2(s3, c3);
 
    % Step 5: shoulder angle, theta2 = alpha - beta
    theta2 = atan2(s, r) - atan2(a_3*s3, a_2 + a_3*c3);
 
    % Step 6: wrist angle
    theta4 = theta_ee - theta2 - theta3;
 
    q = [theta1; theta2; theta3; theta4];
end


%% Jacobian matrix

q = [th1; th2; th3; th4];

% Origins of each coordinate frame (expressed in base frame)
o0 = [0; 0; 0];
T20 = T10 * T21;
T30 = T20 * T32;
o1 = T10(1:3, 4);
o2 = T20(1:3, 4);
o3 = T30(1:3, 4);
o_eff = T_total_stylus(1:3, 4);
o_cam = T_total_camera(1:3, 4);

% Z-axes of each frame (z_{i-1} for i = 1, 2, 3, 4)
z0 = [0; 0; 1]; % Base frame z-axis
z1 = T10(1:3, 3);
z2 = T20(1:3, 3);
z3 = T30(1:3, 3);

%% Build Geometric Jacobians (Revolute joints)

% Stylus End-Effector Jacobian
J_v_eff = [cross(z0, o_eff - o0), cross(z1, o_eff - o1), cross(z2, o_eff - o2), cross(z3, o_eff - o3)];
J_w_eff = [z0, z1, z2, z3];
J_eff = simplify([J_v_eff; J_w_eff])

% Camera Jacobian
J_v_cam = [cross(z0, o_cam - o0), cross(z1, o_cam - o1), cross(z2, o_cam - o2), cross(z3, o_cam - o3)];
J_w_cam = [z0, z1, z2, z3];
J_cam = simplify([J_v_cam; J_w_cam])

%% Numerical evaluation at specific phi values

phi_values = [0, pi/2, pi, 3*pi/2];
