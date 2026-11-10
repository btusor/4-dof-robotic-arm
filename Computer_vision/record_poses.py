"""Record the real joint angles of hand-guided poses: home, then the stylus
pointing at / touching each red Smartie.

    python record_poses.py             # no camera: just record the poses
    python record_poses.py --capture   # first move to the recording pose, take a picture,
                                       # detect the Smarties and number them, then record

Order of the steps: HOME, SMARTIE 1 .. N (numbering = the numbers on the picture
Debug/<run>/06_circles.png that is printed at the start).
Keys while a step is shown:  Enter = record the current angles,  s = skip,  q = quit.
Result: Data/recorded_poses.json (angles q in the move_j convention, image circles,
the measured recording pose and the base-frame position the vision gives for every Smartie).
For a calibration put 3-4 Smarties spread over the table (not in a row) and touch the
CENTRE of each with the stylus tip.

With --capture the arm goes limp after the picture is taken. HOLD IT.
"""
import argparse
import csv
import json
import msvcrt
import os
import time
import numpy as np

from recording import MyRobot, PortHandler, PacketHandler
import kinematics as kin

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_FILE = os.path.join(HERE, "Data", "recorded_poses.json")
ADDR_TORQUE_ENABLE, ADDR_PRESENT_POSITION = 24, 36


def read_q(packet, port):
    q = []
    for i in (1, 2, 3, 4):
        ticks, comm, _ = packet.read2ByteTxRx(port, i, ADDR_PRESENT_POSITION)
        if comm != 0:
            return None
        q.append(ticks * MyRobot.DEG_PER_TICK - 150.0)
    return q


def capture_and_detect():
    """Move to the recording pose, take a picture, detect the Smarties. Returns (image path, circles)."""
    import cv2
    import main
    from camera import open_cam, capture_frame
    cap = open_cam()
    robot = MyRobot()
    if not robot.init_status:
        cap.release()
        raise SystemExit("Robot did not initialise")
    try:
        robot.move_j(*main.HOME_POSE)
        robot.move_j(*main.RECORDING_POSE)
        img = capture_frame(cap)
        q_rec = [float(v) for v in robot.read_joint_angles()]
    finally:
        cap.release()
        cv2.destroyAllWindows()
        input("The arm will go limp now. HOLD IT, then press Enter... ")
        robot.disable_motors()
    run_dir = main.make_run_dir()
    path = os.path.join(main.DATA_DIR, time.strftime("capture_%Y%m%d_%H%M%S.png"))
    cv2.imwrite(path, img)
    points_base = []
    try:
        points_base = main.process_image(img, run_dir, q_rec)
    except ValueError as e:
        print("process_image:", e)
    circles = []
    csv_path = os.path.join(run_dir, "circles.csv")
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            circles = [[float(v) for v in row[1:]] for row in list(csv.reader(f))[1:]]
    print(f"\nPicture: {path}\nNumbered Smarties: {os.path.join(run_dir, '06_circles.png')}")
    for i, (u, v, r) in enumerate(circles):
        print(f"  Smartie {i + 1}: pixel ({u:.0f}, {v:.0f})")
    return path, circles, q_rec, [[float(v) for v in p] for p in points_base]


def main_record(n_smarties, capture):
    result = {"recorded": time.strftime("%Y-%m-%d %H:%M:%S"), "poses": {}}
    if capture:
        result["image"], result["circles_uvr"], result["q_recording_measured"], result["vision_base_xyz"] = capture_and_detect()
        n_smarties = len(result["circles_uvr"]) or n_smarties

    port, packet = PortHandler(MyRobot.DEVICENAME), PacketHandler(MyRobot.PROTOCOL_VERSION)
    if not (port.openPort() and port.setBaudRate(MyRobot.BAUDRATE)):
        raise SystemExit(f"Cannot open {MyRobot.DEVICENAME}")
    if not capture:
        input("The arm will go limp. HOLD IT, then press Enter... ")
    for i in (1, 2, 3, 4):
        packet.write1ByteTxRx(port, i, ADDR_TORQUE_ENABLE, 0)

    steps = ["HOME (arm upright / your home pose)"] + [f"SMARTIE {k}: stylus tip touching it, pointing down" for k in range(1, n_smarties + 1)]
    names = ["home"] + [f"smartie_{k}" for k in range(1, n_smarties + 1)]
    try:
        for name, text in zip(names, steps):
            print(f"\n>>> {text}\n    Enter = record, s = skip, q = quit")
            while True:
                q = read_q(packet, port)
                if q is not None:
                    T = kin.forward_stylus(q)
                    print(f"\r    q = {np.round(q, 1)}  model tip = {np.round(T[:3, 3], 0)}      ", end="", flush=True)
                if msvcrt.kbhit():
                    key = msvcrt.getwch()
                    if key in ("\r", "\n") and q is not None:
                        result["poses"][name] = [round(v, 2) for v in q]
                        print(f"\n    recorded {name}: {result['poses'][name]}")
                        break
                    if key.lower() == "s":
                        print("\n    skipped")
                        break
                    if key.lower() == "q":
                        raise KeyboardInterrupt
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        port.closePort()
        os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
        with open(OUT_FILE, "w") as f:
            json.dump(result, f, indent=2)
        print("Saved", OUT_FILE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="store_true", help="take a picture from the recording pose first")
    parser.add_argument("--n", type=int, default=3, help="number of Smarties (without --capture)")
    args = parser.parse_args()
    main_record(args.n, args.capture)
