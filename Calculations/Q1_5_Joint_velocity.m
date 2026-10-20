clear all;
close all;
clc;

% DH parameters [mm]
d_1 = 50; 
a_2 = 93; 
a_3 = 93; 
a_4 = 50;

% Point on the circle at phi = pi/2, stylus horizontal
p_c = [150; 0; 120];  
R = 32;  
phi = pi/2;

% o_40 = [150; 0; 152]
o_40 = p_c + R*[0; cos(phi); sin(phi)];
x_z0 = 0;

% Elbow-up Inverse Kinematics
q = inverseKinematics(o_40, x_z0, d_1, a_2, a_3, a_4);   

% Geometric Jacobian of the stylus tip, J = [Jv; Jw] (6x4)
J = jacobianStylus(q, d_1, a_2, a_3, a_4);
Jv = J(1:3, :);  Jw = J(4:6, :);

% x-axis of frame 4 (stylus direction) in frame 0
x_4 = [cos(q(1))*cos(q(2)+q(3)+q(4));
       sin(q(1))*cos(q(2)+q(3)+q(4));
       sin(q(2)+q(3)+q(4))];

% Constraint xdot_4,z = 0, with xdot_4 = w x x_4  ->  [x_4y, -x_4x, 0]*Jw*qdot = 0
c = [x_4(2), -x_4(1), 0] * Jw;

% Square 4x4 system  A*qdot = b
A = [Jv; c];
b = [0; -3; 0; 0];
qdot = A \ b;                                      % [rad/s]

% Check: resulting tip velocity, angular velocity and xdot_4
v      = Jv*qdot;
w      = Jw*qdot;
xdot_4 = cross(w, x_4);

fprintf('q    = [%7.2f %7.2f %7.2f %7.2f] deg\n', rad2deg(q));
fprintf('qdot = [%7.4f %7.4f %7.4f %7.4f] rad/s\n', qdot);
fprintf('qdot = [%7.4f %7.4f %7.4f %7.4f] deg/s\n', rad2deg(qdot));
fprintf('v    = [%7.4f %7.4f %7.4f] mm/s\n', v);
fprintf('w    = [%7.4f %7.4f %7.4f] rad/s\n', w);
fprintf('xdot_4 = [%7.4f %7.4f %7.4f] 1/s\n', xdot_4);
fprintf('cond(A) = %.2f\n', cond(A));

%% Local functions

function q = inverseKinematics(o_40, x_z0, d_1, a_2, a_3, a_4)
    q1   = atan2(o_40(2), o_40(1));
    thee = asin(x_z0);
    r    = sqrt(o_40(1)^2 + o_40(2)^2) - a_4*cos(thee);
    s    = o_40(3) - d_1 - a_4*sin(thee);
    c3   = (r^2 + s^2 - a_2^2 - a_3^2)/(2*a_2*a_3);
    q3   = atan2(-sqrt(1 - c3^2), c3);              
    q2   = atan2(s, r) - atan2(a_3*sin(q3), a_2 + a_3*cos(q3));
    q4   = thee - q2 - q3;
    q    = [q1; q2; q3; q4];
end

function J = jacobianStylus(q, d_1, a_2, a_3, a_4)
    % Columns: Jv_i = z_{i-1} x (o_4 - o_{i-1}),  Jw_i = z_{i-1}
    DH = [q(1) d_1 0   pi/2;
          q(2) 0   a_2 0;
          q(3) 0   a_3 0;
          q(4) 0   a_4 0];
    T = eye(4);
    z = zeros(3, 5);  o = zeros(3, 5);
    z(:, 1) = [0; 0; 1];
    for i = 1:4
        th = DH(i,1); d = DH(i,2); a = DH(i,3); al = DH(i,4);
        A = [cos(th) -sin(th)*cos(al)  sin(th)*sin(al) a*cos(th);
             sin(th)  cos(th)*cos(al) -cos(th)*sin(al) a*sin(th);
             0        sin(al)          cos(al)         d;
             0        0                0               1];
        T = T*A;
        z(:, i+1) = T(1:3, 3);
        o(:, i+1) = T(1:3, 4);
    end
    J = zeros(6, 4);
    for i = 1:4
        J(1:3, i) = cross(z(:, i), o(:, 5) - o(:, i));
        J(4:6, i) = z(:, i);
    end
end
