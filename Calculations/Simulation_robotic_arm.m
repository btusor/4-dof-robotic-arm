clear all;
close all;
clc;

%% Robot geometry Parameters

l_1 = 50; %[mm]
l_2 = 93; %[mm]
l_3 = 93; %[mm]
l_4 = 50; %[mm]
l_5 = 45; %[mm]  
l_6 = 35; %[mm]  

%% DH parameters [d, a, alpha] for each joint (theta is the joint variable)

DH = [
    l_1, 0,   pi/2;
    0,   l_2, 0;
    0,   l_3, 0;
    0,   l_4, 0
    ];

theta_offset = [0, pi/2, 0, 0]; % [rad]

% Transformation from frame {4} (stylus) to frame {5} (camera):
T54 = [
    1,  0,  0, -(l_4 - l_6);
    0,  1,  0,  l_5;
    0,  0,  1,  0;
    0,  0,  0,  1
    ];

%% Build the GUI

robot.DH = DH;
robot.theta_offset = theta_offset;
robot.T54 = T54;
robot.l_4 = l_4;
robot.l_6 = l_6;

fig = figure('Name', 'Robot arm simulation - Group 13', 'NumberTitle', 'off', ...
    'Color', 'w', 'Position', [100 100 1100 700]);

ax = axes('Parent', fig, 'Units', 'normalized', 'Position', [0.05 0.08 0.62 0.88]);
hold(ax, 'on'); grid(ax, 'on'); axis(ax, 'equal');
xlabel(ax, 'x_0 [mm]'); ylabel(ax, 'y_0 [mm]'); zlabel(ax, 'z_0 [mm]');
reach = l_1 + l_2 + l_3 + l_4;
axis(ax, [-reach reach -reach reach -reach/2 reach+20]);
view(ax, 35, 20);

% Table / base block under frame {0}
[Xb, Yb] = meshgrid([-60 60], [-60 60]);
surf(ax, Xb, Yb, zeros(size(Xb)), 'FaceColor', [0.3 0.3 0.3], ...
    'FaceAlpha', 0.4, 'EdgeColor', 'none');

robot.ax = ax;
robot.h = [];

names = {'q_1', 'q_2', 'q_3', 'q_4'};
for i = 1:4
    y = 0.85 - (i-1)*0.12;
    uicontrol(fig, 'Style', 'text', 'Units', 'normalized', ...
        'Position', [0.70 y+0.04 0.10 0.04], 'String', sprintf('Joint %d', i), ...
        'BackgroundColor', 'w', 'FontSize', 11, 'HorizontalAlignment', 'left');
    robot.slider(i) = uicontrol(fig, 'Style', 'slider', 'Units', 'normalized', ...
        'Position', [0.70 y 0.20 0.04], 'Min', -90, 'Max', 90, 'Value', 0, ...
        'SliderStep', [1/180 10/180], 'Callback', @(src, evt) update_robot(fig));
    robot.label(i) = uicontrol(fig, 'Style', 'edit', 'Units', 'normalized', ...
        'Position', [0.91 y 0.06 0.04], 'String', '0', 'FontSize', 11, ...
        'Callback', @(src, evt) edit_changed(fig, i));
    % Live update while dragging
    addlistener(robot.slider(i), 'Value', 'PostSet', @(src, evt) update_robot(fig));
end

uicontrol(fig, 'Style', 'pushbutton', 'Units', 'normalized', ...
    'Position', [0.70 0.30 0.08 0.05], 'String', 'Reset', 'FontSize', 11, ...
    'Callback', @(src, evt) reset_robot(fig));
uicontrol(fig, 'Style', 'pushbutton', 'Units', 'normalized', ...
    'Position', [0.79 0.30 0.09 0.05], 'String', 'Front view', 'FontSize', 11, ...
    'Callback', @(src, evt) view(ax, 0, 0));
uicontrol(fig, 'Style', 'pushbutton', 'Units', 'normalized', ...
    'Position', [0.89 0.30 0.08 0.05], 'String', '3D view', 'FontSize', 11, ...
    'Callback', @(src, evt) view(ax, 35, 20));

robot.info = uicontrol(fig, 'Style', 'text', 'Units', 'normalized', ...
    'Position', [0.70 0.05 0.28 0.22], 'BackgroundColor', 'w', ...
    'FontName', 'Consolas', 'FontSize', 10, 'HorizontalAlignment', 'left');

guidata(fig, robot);
update_robot(fig);

%% Local functions

% General Denavit-Hartenberg transformation matrix
function T = dh_trans(theta, d, a, alpha)
T = [
    cos(theta), -sin(theta)*cos(alpha),  sin(theta)*sin(alpha), a*cos(theta);
    sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta);
    0,           sin(alpha),             cos(alpha),            d;
    0,           0,                      0,                     1
    ];
end

function T = forward_kinematics(robot, q_deg)
% Returns T(:,:,k) = transformation from frame {k-1} to frame {0}, k = 1..6
% i.e. frames {0},{1},{2},{3},{4},{5}
theta = deg2rad(q_deg) + robot.theta_offset;
T = zeros(4, 4, 6);
T(:,:,1) = eye(4);
for i = 1:4
    T(:,:,i+1) = T(:,:,i) * dh_trans(theta(i), robot.DH(i,1), robot.DH(i,2), robot.DH(i,3));
end
T(:,:,6) = T(:,:,5) * robot.T54;
end

function update_robot(fig)
robot = guidata(fig);
if isempty(robot) || ~isfield(robot, 'slider')
    return
end
q = arrayfun(@(s) round(get(s, 'Value')), robot.slider);
for i = 1:4
    set(robot.label(i), 'String', sprintf('%d', q(i)));
end

T = forward_kinematics(robot, q);
o = squeeze(T(1:3, 4, :)); % origins of frames {0}..{5}, one per column

ax = robot.ax;
delete(robot.h);
h = gobjects(0);

% Links: base -> {1} -> {2} -> {3} -> {4} (stylus tip)
h(end+1) = plot3(ax, o(1,1:5), o(2,1:5), o(3,1:5), '-', ...
    'Color', [0.55 0.55 0.6], 'LineWidth', 10);
% Camera bracket: from the stylus axis at the camera height out to {5}
x4 = T(1:3, 1, 5);
p_bracket = o(:,5) - (robot.l_4 - robot.l_6) * x4;
h(end+1) = plot3(ax, [p_bracket(1) o(1,6)], [p_bracket(2) o(2,6)], ...
    [p_bracket(3) o(3,6)], '-', 'Color', [0.35 0.35 0.4], 'LineWidth', 5);
% Joints
h(end+1) = plot3(ax, o(1,2:4), o(2,2:4), o(3,2:4), 'o', 'MarkerSize', 10, ...
    'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k');

% Frame axes: x red, y green, z blue
L = 30;
cols = {'r', [0 0.6 0], 'b'};
for k = 1:6
    for j = 1:3
        d = T(1:3, j, k) * L;
        h(end+1) = quiver3(ax, o(1,k), o(2,k), o(3,k), d(1), d(2), d(3), 0, ...
            'Color', cols{j}, 'LineWidth', 2, 'MaxHeadSize', 0.5);
    end
    h(end+1) = text(ax, o(1,k) + 8, o(2,k) + 8, o(3,k) + 8, sprintf('\\{%d\\}', k-1), ...
        'FontSize', 11, 'FontWeight', 'bold');
end

robot.h = h;
guidata(fig, robot);

p4 = o(:,5); p5 = o(:,6);
set(robot.info, 'String', { ...
    sprintf('q = [%d  %d  %d  %d] deg', q), ...
    '', ...
    'Stylus tip {4} in {0} [mm]:', ...
    sprintf('  x = %7.1f  y = %7.1f  z = %7.1f', p4), ...
    '', ...
    'Camera {5} in {0} [mm]:', ...
    sprintf('  x = %7.1f  y = %7.1f  z = %7.1f', p5)});
drawnow limitrate
end

function edit_changed(fig, i)
robot = guidata(fig);
v = str2double(get(robot.label(i), 'String'));
if isnan(v)
    v = get(robot.slider(i), 'Value');
end
set(robot.slider(i), 'Value', max(-90, min(90, v)));
update_robot(fig);
end

function reset_robot(fig)
robot = guidata(fig);
for i = 1:4
    set(robot.slider(i), 'Value', 0);
end
update_robot(fig);
end
