%% Problem 10 - Inverse Dynamics 
% Denavit-Hartenberg parameters of the 4 DoF robotic arm
% \\ Theta_i | d_i | a_i | alpha_i
% 1. theta_1 | 50  |  0  | pi/2
% 2. theta_2 |  0  |  93 |   0
% 3. theta_3 |  0  |  93 |   0
% 4. theta_4 |  0  |  50 |   0

clear;
close all;
clc;

syms theta_1 theta_2 theta_3 theta_4 real

theta = [theta_1, theta_2, theta_3, theta_4];
d = [50, 0, 0, 0];
a = [0, 93, 93, 50];
alpha = [sym(pi)/2, 0, 0, 0];

DH = @(th, d, a, al) [ cos(th), -sin(th)*cos(al),  sin(th)*sin(al), a*cos(th);
                        sin(th),  cos(th)*cos(al), -cos(th)*sin(al), a*sin(th);
                              0,           sin(al),          cos(al),        d;
                              0,                 0,                0,        1];

T01 = DH(theta(1), d(1), a(1), alpha(1));
T12 = DH(theta(2), d(2), a(2), alpha(2));
T23 = DH(theta(3), d(3), a(3), alpha(3));
T34 = DH(theta(4), d(4), a(4), alpha(4));

T45 = [1 0 0 -15;
       0 1 0  45;
       0 0 1  0;
       0 0 0  1];

T02 = simplify(T01 * T12);
T03 = simplify(T02 * T23);
T04 = simplify(T03 * T34);
T05 = simplify(T04 * T45);

% Robot properties:
m1 = 60;
m2 = 80;
m3 = m2;
m4 = 40;
a = 32;     % Assumed dimensions of link 1: 32x40x50
b = 40;
c = 50;
I0 = 20500; % g*mm^2 estimated (2*10^-5 kg*m^2)

%syms m1 m2 m3 m4 I0 a b c

% matrices of inertia
D1 = [I0 0 0;
      0 0.4*I0 0;
      0 0 0.9*I0];
D2 = [0.45*I0 0 0;
      0 1.4*I0 0;
      0 0 1.2*I0];
D3 = D2;
D4 = [0.5*I0 0 0;
      0 0.5*I0 0
      0 0 0.5*I0];

% positions for each center of mass
rc1 = T01(1:3,4) + T01(1:3,1:3)*[0; -20; 0];
rc2 = T02(1:3,4) + T02(1:3,1:3)*[-30; 0; 0];
rc3 = T03(1:3,4) + T03(1:3,1:3)*[-30; 0; 0];
rc4 = T04(1:3,4) + T04(1:3,1:3)*[-25; 15; 0];

% z axes
z0 = [0; 0; 1;];
z1 = T01(1:3, 3);
z2 = T02(1:3, 3);
z3 = T03(1:3, 3);

% origins
o0 = [0; 0; 0];
o1 = T01(1:3, 4);
o2 = T02(1:3, 4);
o3 = T03(1:3, 4);
o4 = T04(1:3, 4);
o5 = T05(1:3, 4);

% Velocity Jacobians
Jv1 = simplify([cross(z0, rc1 - o0), zeros(3,3)]);
Jv2 = simplify([cross(z0, rc2 - o0), cross(z1, rc2 - o1), zeros(3,2)]);
Jv3 = simplify([cross(z0, rc3 - o0), cross(z1, rc3 - o1), cross(z2, rc3 - o2), zeros(3,1)]);
Jv4 = simplify([cross(z0, rc4 - o0), cross(z1, rc4 - o1), cross(z2, rc4 - o2), cross(z3, rc4 - o3)]);
% Angular Jacobians
Jw1 = [z0, zeros(3,3)];
Jw2 = [z0, z1, zeros(3,2)];
Jw3 = [z0, z1, z2, zeros(3,1)];
Jw4 = [z0, z1, z2, z3];

% Inertia tensors in cubic form for link 1
Ixx = m1/12*(b^2+c^2);
Iyy = m1/12*(a^2+b^2);
Izz = m1/12*(a^2+c^2);
% Based on the values above I'd estimate I0 to 20000 g*mm^2
% or 2*1e-5 kg*m^2

% Rotating the inertia tensors to the world frames
R01 = T01(1:3,1:3);
R02 = T02(1:3,1:3);
R03 = T03(1:3,1:3);
R04 = T04(1:3,1:3);

I1 = simplify(R01 * D1 * R01.');
I2 = simplify(R02 * D2 * R02.');
I3 = simplify(R03 * D3 * R03.');
I4 = simplify(R04 * D4 * R04.');

% Calculating the inertia matrix
D = simplify(m1*(Jv1.'*Jv1) + Jw1.'*I1*Jw1 + ...
             m2*(Jv2.'*Jv2) + Jw2.'*I2*Jw2 + ...
             m3*(Jv3.'*Jv3) + Jw3.'*I3*Jw3 + ...
             m4*(Jv4.'*Jv4) + Jw4.'*I4*Jw4);

%% Christoffel symbols  c(i,j,k) = 1/2*(dD_kj/dq_i + dD_ki/dq_j - dD_ij/dq_k)
n = 4;
c = sym(zeros(n,n,n));
for k = 1:n
    for i = 1:n
        for j = 1:n
            c(i,j,k) = 1/2 * ( diff(D(k,j), theta(i)) ...
                             + diff(D(k,i), theta(j)) ...
                             - diff(D(i,j), theta(k)) );
        end
    end
end

% Coriolis / centrifugal matrix  C(k,j) = sum_i c(i,j,k) * dq_i
syms dtheta_1 dtheta_2 dtheta_3 dtheta_4 real
dtheta = [dtheta_1, dtheta_2, dtheta_3, dtheta_4];
C = sym(zeros(n,n));
for k = 1:n
    for j = 1:n
        for i = 1:n
            C(k,j) = C(k,j) + c(i,j,k) * dtheta(i);
        end
    end
end
C = simplify(C);

% Check 1: D must not depend on theta_1
disp(simplify(diff(D, theta_1)));          % should be all zeros

% Check 2: Ddot - 2C must be skew-symmetric
Ddot = sym(zeros(n,n));
for i = 1:n
    Ddot = Ddot + diff(D, theta(i)) * dtheta(i);
end
N = Ddot - 2*C;
disp(simplify(N + N.'));                    % should be all zeros

% Potential Energy
syms g0
g = [0, 0, -g0];
P = -m1*g*rc1-m2*g*rc2-m3*g*rc3-m4*g*rc4;
G = simplify(jacobian(P, theta).');

%% Torque equation  tau = D*qdd + C*qd + G
syms ddtheta_1 ddtheta_2 ddtheta_3 ddtheta_4 real
ddtheta = [ddtheta_1, ddtheta_2, ddtheta_3, ddtheta_4];

tau = D*ddtheta.' + C*dtheta.' + G;
tau = subs(tau, g0, 9810);                 

for k = 1:4
    tk = collect(simplify(tau(k)), [ddtheta, dtheta]);  % group by qdd_j and qd_i*qd_j
    fprintf('\ntau_%d =\n', k);
    disp(vpa(tk, 4));                      % 4 significant digits on the coefficients
end

