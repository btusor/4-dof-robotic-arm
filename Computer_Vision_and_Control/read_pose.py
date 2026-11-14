"""Read the joint angles live while you move the arm by hand.

Prints the joint angles in the move_j convention (motor angle - 150 deg) and what
the kinematic model in kinematics.py predicts for them (stylus tip position and
pitch). Compare with the real arm to find wrong joint zeros / signs / offsets.

    python read_pose.py          # read only, motors are left as they are
    python read_pose.py --free   # also switch the torque OFF so the arm can be moved by hand
                                 # (the arm drops under gravity: hold it!)

Press s to save the current pose as a line in Data/saved_poses.txt.
Stop with the q key (or Ctrl+C). Torque stays off after --free until the next program enables it.
"""
import argparse
import msvcrt
import os
import time
import numpy as np

from recording import MyRobot, PortHandler, PacketHandler
import kinematics as kin

ADDR_TORQUE_ENABLE, ADDR_PRESENT_POSITION = 24, 36
SAVE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "saved_poses.txt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--free", action="store_true", help="torque off on joints 1-4")
    args = parser.parse_args()

    port, packet = PortHandler(MyRobot.DEVICENAME), PacketHandler(MyRobot.PROTOCOL_VERSION)
    if not (port.openPort() and port.setBaudRate(MyRobot.BAUDRATE)):
        raise SystemExit(f"Cannot open {MyRobot.DEVICENAME}")
    ids = (1, 2, 3, 4)
    if args.free:
        input("The arm will go limp. Hold it, then press Enter... ")
        for i in ids:
            packet.write1ByteTxRx(port, i, ADDR_TORQUE_ENABLE, 0)
    print("q = [J1, J2, J3, J4] deg | model: tip x y z [mm], stylus pitch [deg] (-90 = straight down)")
    print("Press s to save the pose to", SAVE_FILE, "| q to quit.")
    n_saved = 0
    try:
        while True:
            key = msvcrt.getwch().lower() if msvcrt.kbhit() else ""
            if key == "q":
                break
            q = []
            for i in ids:
                ticks, comm, _ = packet.read2ByteTxRx(port, i, ADDR_PRESENT_POSITION)
                q.append(np.nan if comm != 0 else ticks * MyRobot.DEG_PER_TICK - 150.0)
            if not np.isnan(q).any():
                T = kin.forward_stylus(q)
                pitch = np.degrees(np.arcsin(T[2, 0]))
                if key == "s":
                    n_saved += 1
                    os.makedirs(os.path.dirname(SAVE_FILE), exist_ok=True)
                    with open(SAVE_FILE, "a") as f:
                        f.write(f"{n_saved}  {time.strftime('%H:%M:%S')}  q={[round(v, 1) for v in q]}  "
                                f"model_tip={np.round(T[:3, 3], 0).tolist()}  model_pitch={pitch:.1f}\n")
                    print(f"\nsaved #{n_saved}: q={np.round(q, 1)}")
                print(f"q = {np.round(q, 1)} | tip = {np.round(T[:3, 3], 0)}  pitch = {pitch:6.1f}", end="\r")
            time.sleep(0.3)
    except KeyboardInterrupt:
        print()
    finally:
        port.closePort()


if __name__ == "__main__":
    main()
