clear all;
close all;
clc;

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