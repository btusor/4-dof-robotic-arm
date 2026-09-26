clc;
clear;

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
T_total_stylus = simplify(T10 * T21 * T32 * T43)

% Transformation matrix between frame 4 and 5
T54 = [
    0,  0,  0, -15;
    0,  1,  0,  45;
    0,  0,  1,  0;
    0,  0,  0,  1
    ];

% Robot camera cumulative forward kinematic model and simplify
T_total_camera = simplify(T_total_stylus * T54)