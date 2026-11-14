"""Servo limit / blockage test for the 4 DoF arm.

Part 1  Servo registers: angle limits, alarm LED/shutdown, torque limit,
        temperature, voltage, hardware error bits.
Part 2  Joint sweeps: every joint is stepped from home towards each end of its
        test range (the others stay at 0). After every step the measured angle,
        load and error bits are read. A joint that does not follow, is in
        overload or raises an alarm is reported and the sweep stops there.
Part 3  (optional, --image) replays the real hover/touch poses that main.py
        computes for the Smarties in an image and checks that every pose is
        reached.

Usage:
    python servo_limit_test.py                       # registers + sweeps
    python servo_limit_test.py --image Data/capture_X.png   # + the real poses
    python servo_limit_test.py --j4 -140 100         # change one joint's range

The arm MOVES. Keep the table clear and a hand near the power switch.
Defaults are limited to ranges that do not drive the arm into the table.
"""
import argparse
import time
import numpy as np

from recording import MyRobot, PortHandler, PacketHandler

HOME = [0, 0, 0, 0]
STEP_DEG = 10
SETTLE_S = 2.5
TRACK_TOL_DEG = 4.0      # |measured - commanded| above this = joint is not following
LOAD_WARN = 600          # present load, 0..1023 = 0..100 % of max torque

# default sweep ranges [deg] (robot convention, 0 = upright)
RANGES = {1: (-90, 90), 2: (-100, 20), 3: (-120, 120), 4: (-120, 100)}

ADDR_CW_LIMIT, ADDR_CCW_LIMIT = 6, 8
ADDR_ALARM_LED, ADDR_ALARM_SHUTDOWN = 17, 18
ADDR_TORQUE_LIMIT = 34
ADDR_LOAD, ADDR_VOLTAGE, ADDR_TEMP = 40, 42, 43
ERROR_BITS = ["input voltage", "angle limit", "overheating", "range", "checksum", "overload", "instruction"]


def decode_error(err):
    return [name for bit, name in enumerate(ERROR_BITS) if err & (1 << bit)]


def read_reg(robot, dxl_id, addr, size):
    """Return (value, hardware_error_byte). value is None if the read failed."""
    if size == 1:
        v, comm, err = robot.packet_handler.read1ByteTxRx(robot.port_handler, dxl_id, addr)
    else:
        v, comm, err = robot.packet_handler.read2ByteTxRx(robot.port_handler, dxl_id, addr)
    return (v if comm == robot.COMM_SUCCESS else None), err


def preflight():
    """Read every servo WITHOUT raising and try to clear a latched alarm.

    MyRobot() aborts on the first servo that answers with an error byte, e.g.
    'DXL write2 error id=4, addr=32, error=32' = overload alarm (bit 5), which
    leaves the servo shut down. Returns the ids that still report an alarm.
    """
    port, packet = PortHandler(MyRobot.DEVICENAME), PacketHandler(MyRobot.PROTOCOL_VERSION)
    if not (port.openPort() and port.setBaudRate(MyRobot.BAUDRATE)):
        raise SystemExit(f"Cannot open {MyRobot.DEVICENAME}")
    class _R:  # minimal object with what read_reg needs
        packet_handler, port_handler, COMM_SUCCESS = packet, port, MyRobot.COMM_SUCCESS
    alarmed = []
    print("== Preflight ==")
    for dxl_id in (1, 2, 3, 4):
        for attempt in (1, 2):
            temp, err = read_reg(_R, dxl_id, ADDR_TEMP, 1)
            load, _ = read_reg(_R, dxl_id, ADDR_LOAD, 2)
            volt, _ = read_reg(_R, dxl_id, ADDR_VOLTAGE, 1)
            pos, _ = read_reg(_R, dxl_id, 36, 2)
            print(f"servo {dxl_id}: pos={pos} load={None if load is None else load & 0x3FF} temp={temp}C "
                  f"volt={None if volt is None else volt / 10}V alarm={decode_error(err) or 'none'}")
            if not err or attempt == 2:
                break
            # try to clear: torque off, then on again
            packet.write1ByteTxRx(port, dxl_id, 24, 0)
            time.sleep(0.3)
            packet.write1ByteTxRx(port, dxl_id, 24, 1)
            time.sleep(0.3)
        if err:
            alarmed.append(dxl_id)
    port.closePort()
    return alarmed


def servo_report(robot):
    print("\n== Servo registers ==")
    print("id  angle limits [deg, motor]   joint range [deg]   alarm LED/shutdown   torque limit   temp   volt")
    for i, dxl_id in enumerate(robot.motor_ids):
        cw, _ = read_reg(robot, dxl_id, ADDR_CW_LIMIT, 2)
        ccw, _ = read_reg(robot, dxl_id, ADDR_CCW_LIMIT, 2)
        led, _ = read_reg(robot, dxl_id, ADDR_ALARM_LED, 1)
        shut, _ = read_reg(robot, dxl_id, ADDR_ALARM_SHUTDOWN, 1)
        tq, _ = read_reg(robot, dxl_id, ADDR_TORQUE_LIMIT, 2)
        temp, err = read_reg(robot, dxl_id, ADDR_TEMP, 1)
        volt, _ = read_reg(robot, dxl_id, ADDR_VOLTAGE, 1)
        if cw is None or ccw is None:
            print(f"{dxl_id}   no answer from the servo")
            continue
        lo, hi = cw * robot.DEG_PER_TICK, ccw * robot.DEG_PER_TICK
        off = robot.joint_offsets[i]
        print(f"{dxl_id}   {lo:6.1f} .. {hi:6.1f}            {lo - off:6.1f} .. {hi - off:6.1f}      "
              f"LED={led} shutdown={shut}        {tq}          {temp}C   {volt / 10 if volt else None}V"
              + (f"   ERROR: {decode_error(err)}" if err else ""))


def wait_settled(robot, target, timeout=SETTLE_S):
    """Wait until the joints reach target or stop moving. Returns the last reading."""
    t_end = time.time() + timeout
    last = None
    while time.time() < t_end:
        try:
            last = robot.read_joint_angles()
        except Exception:
            time.sleep(0.05)
            continue
        if np.all(np.abs(last - target) < 1.5):
            break
        time.sleep(0.1)
    return last


def check_joint(robot, idx, target):
    """Return a list of problems on joint idx after a move, empty if all is fine."""
    problems = []
    dxl_id = robot.motor_ids[idx]
    measured = wait_settled(robot, target)
    if measured is None:
        return ["no reading"], None, None
    err_deg = measured[idx] - target[idx]
    load, herr = read_reg(robot, dxl_id, ADDR_LOAD, 2)
    load_pct = None if load is None else (load & 0x3FF)
    if abs(err_deg) > TRACK_TOL_DEG:
        problems.append(f"not following ({err_deg:+.1f} deg)")
    if load_pct is not None and load_pct > LOAD_WARN:
        problems.append(f"load {load_pct}/1023")
    if herr:
        problems.append("alarm: " + ", ".join(decode_error(herr)))
    return problems, err_deg, load_pct


def sweep(robot, idx, lo, hi):
    """Step joint idx (0-based) to both ends of [lo, hi]. Returns {'neg': deg, 'pos': deg} reached."""
    result = {}
    for name, end in (("neg", lo), ("pos", hi)):
        sign = -1 if end < 0 else 1
        reached = 0
        print(f"\nJoint {idx + 1}, towards {end:+d} deg")
        for deg in range(sign * STEP_DEG, end + sign, sign * STEP_DEG):
            target = np.array(HOME, dtype=float)
            target[idx] = deg
            try:
                robot.move_j(*target, wait=False)
            except ValueError as e:
                print(f"  {deg:+4d}: software limit ({e})")
                break
            problems, err, load = check_joint(robot, idx, target)
            print(f"  {deg:+4d}: err={err if err is None else round(err, 1):>6}  load={load}  "
                  + ("OK" if not problems else "PROBLEM: " + "; ".join(problems)))
            if problems:
                break
            reached = deg
        result[name] = reached
        robot.move_j(*HOME, wait=False)
        wait_settled(robot, np.array(HOME, dtype=float), timeout=4)
    return result


def replay_poses(robot, image_path):
    """Replay the hover/touch poses main.py computes for the Smarties in the image."""
    import cv2
    import main
    img = cv2.imread(image_path)
    if img is None:
        raise SystemExit(f"Could not read {image_path}")
    import tempfile
    points = main.process_image(img, tempfile.mkdtemp())
    print("\n== Real poses from", image_path, "==")
    for i, p in enumerate(sorted(points, key=lambda p: np.hypot(p[0], p[1]))):
        try:
            q_hover, q_touch = main.touch_targets(p)
        except ValueError as e:
            print(f"Smartie #{i+1}: {e}")
            continue
        for name, q in (("hover", q_hover), ("touch", q_touch)):
            target = np.array(q, dtype=float)
            print(f"Smartie #{i+1} {name}: {np.round(target, 1)}")
            robot.move_j(*HOME, wait=False)
            wait_settled(robot, np.array(HOME, dtype=float), timeout=4)
            robot.move_j(*target, wait=False)
            measured = wait_settled(robot, target, timeout=5)
            if measured is None:
                print("   no reading")
                continue
            err = measured - target
            bad = [j + 1 for j in range(4) if abs(err[j]) > TRACK_TOL_DEG]
            print(f"   error per joint [deg]: {np.round(err, 1)}  " + (f"-> joint(s) {bad} NOT reached" if bad else "OK"))
            for j, dxl_id in enumerate(robot.motor_ids):
                _, herr = read_reg(robot, dxl_id, ADDR_TEMP, 1)
                if herr:
                    print(f"   servo {dxl_id} alarm: {decode_error(herr)}")
    robot.move_j(*HOME, wait=False)
    wait_settled(robot, np.array(HOME, dtype=float), timeout=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="also replay the real Smartie poses computed from this image")
    parser.add_argument("--speed", type=float, default=0.1, help="motor speed 0..1 (default 0.1)")
    for j in range(1, 5):
        parser.add_argument(f"--j{j}", nargs=2, type=int, metavar=("MIN", "MAX"), help=f"sweep range of joint {j}")
    parser.add_argument("--skip-sweeps", action="store_true")
    args = parser.parse_args()
    for j in range(1, 5):
        if getattr(args, f"j{j}"):
            RANGES[j] = tuple(getattr(args, f"j{j}"))

    alarmed = preflight()
    if alarmed:
        raise SystemExit(f"Servo(s) {alarmed} still report an alarm (overload = the joint was blocked or "
                         "driven against a stop). Cut the 12 V power for a few seconds, free the arm by hand, "
                         "power on again and restart. Check the temperature above: let it cool if it is hot.")
    robot = MyRobot()
    if not robot.init_status:
        raise SystemExit("Robot did not initialise (port / power?)")
    try:
        robot.set_speed([args.speed] * 4, overwrite_speeds=True)
        servo_report(robot)
        summary = {}
        if not args.skip_sweeps:
            for j in range(1, 5):
                lo, hi = RANGES[j]
                summary[j] = sweep(robot, j - 1, lo, hi)
            print("\n== Summary (deg reached without a problem, tested range in brackets) ==")
            for j, r in summary.items():
                lo, hi = RANGES[j]
                print(f"Joint {j}: {r['neg']:+4d} .. {r['pos']:+4d}   [{lo:+d} .. {hi:+d}]"
                      + ("   <-- stopped early" if r['neg'] != (lo // STEP_DEG) * STEP_DEG or r['pos'] != (hi // STEP_DEG) * STEP_DEG else ""))
        if args.image:
            replay_poses(robot, args.image)
    finally:
        try:
            robot.move_j(*HOME, wait=False)
            time.sleep(2)
        except Exception:
            pass
        robot.disable_motors()
