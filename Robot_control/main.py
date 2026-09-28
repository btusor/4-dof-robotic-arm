import time
import dynamixel_sdk as dxl

# Constants and address definitions
ADDR_MX_TORQUE_ENABLE        = 24
ADDR_MX_CW_COMPLIANCE_MARGIN = 26
ADDR_MX_CCW_COMPLIANCE_MARGIN = 27
ADDR_MX_CW_COMPLIANCE_SLOPE  = 28
ADDR_MX_CCW_COMPLIANCE_SLOPE = 29
ADDR_MX_GOAL_POSITION        = 30
ADDR_MX_MOVING_SPEED         = 32
ADDR_MX_PRESENT_POSITION     = 36
ADDR_MX_PUNCH                = 48

PROTOCOL_VERSION             = 1.0
DXL_IDS                      = [1, 2, 3, 4]  # The 4 servos of the 4-DOF robot arm
DEVICENAME                   = 'COM3'        # Check the actual port name in Dynamixel Wizard
BAUDRATE                     = 1000000

TORQUE_ENABLE                = 1
TORQUE_DISABLE               = 0

# Initialization and port opening
portHandler = dxl.PortHandler(DEVICENAME)
packetHandler = dxl.PacketHandler(PROTOCOL_VERSION)

# Open port
if portHandler.openPort():
    print(f"Successfully opened the port: {DEVICENAME}")
else:
    print(f"Error: Failed to open the port ({DEVICENAME}).")
    quit()

# Set baudrate
if portHandler.setBaudRate(BAUDRATE):
    print(f"Baudrate set to: {BAUDRATE}")
else:
    print("Error: Failed to set the baudrate.")
    quit()

# Servo configuration and torque enable
for DXL_ID in DXL_IDS:
    # Enable torque
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(
        portHandler, DXL_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE
    )
    if dxl_comm_result != dxl.COMM_SUCCESS:
        print(f"[ID {DXL_ID}] Error (Torque Enable): {packetHandler.getTxRxResult(dxl_comm_result)}")
    elif dxl_error != 0:
        print(f"[ID {DXL_ID}] Error (Torque Enable - DXL): {packetHandler.getRxPacketError(dxl_error)}")

    # Set Compliance Margin (0 = no deadband / error margin)
    # These are 1-byte registers on the AX-12A; a 2-byte write to address 27 would also overwrite address 28
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_CW_COMPLIANCE_MARGIN, 0)
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_CCW_COMPLIANCE_MARGIN, 0)

    # Set Compliance Slope (32 = default softer/stiffer characteristic)
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_CW_COMPLIANCE_SLOPE, 32)
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_CCW_COMPLIANCE_SLOPE, 32)

    # Set default moving speed
    packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_MOVING_SPEED, 100)

print("Robot arm initialization and configuration complete.")

# Helper functions for movement and position reading
def move_robot(target_positions, speed=100):
    """
    Moves the 4 servos to the specified target positions.
    target_positions: list [pos1, pos2, pos3, pos4]
    """
    for idx, DXL_ID in enumerate(DXL_IDS):
        # AX-12A position range: 0-1023 (0-300 deg, 512 = 150 deg = center)
        goal_pos = max(0, min(1023, int(target_positions[idx])))

        # Set speed before movement (optional, if speed adjustments are needed per waypoint)
        packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_MOVING_SPEED, speed)

        # Send goal position (2-byte data for AX-12A servos)
        dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(
            portHandler, DXL_ID, ADDR_MX_GOAL_POSITION, goal_pos
        )
        if dxl_comm_result != dxl.COMM_SUCCESS:
            print(f"[ID {DXL_ID}] Movement error: {packetHandler.getTxRxResult(dxl_comm_result)}")

def read_present_positions():
    """Reads and returns the current position of all joints."""
    positions = []
    for DXL_ID in DXL_IDS:
        pos, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(
            portHandler, DXL_ID, ADDR_MX_PRESENT_POSITION
        )
        if dxl_comm_result == dxl.COMM_SUCCESS:
            positions.append(pos)
        else:
            positions.append(None)
    return positions


# main program
try:
    # Move to home position
    home_position = [512, 512, 512, 512]  # 512 is the center position on the 0-1023 scale
    print(f"Moving to home position: {home_position}")
    move_robot(home_position, speed=80)
    time.sleep(2.0)  # Allow time for movement to complete

    # Test waypoint sequence
    # Small excursions (max ~80 ticks = ~23 deg from center) for the first test,
    # since it is not yet verified that 512 corresponds to the upright pose in Figure 1.
    # Increase them once the zero offsets and directions of the joints are known.
    waypoints = [
        [512, 470, 560, 512],
        [440, 490, 540, 460],
        [590, 450, 480, 560],
        [512, 512, 512, 512]   # Return to center
    ]

    for i, wp in enumerate(waypoints):
        print(f"[{i+1}. Step] Target positions: {wp}")
        move_robot(wp, speed=120)
        
        # Wait for servos to reach the target (or simple delay)
        time.sleep(1.5)
        
        # Check current positions
        current_pos = read_present_positions()
        print(f"   -> Current positions: {current_pos}")

except KeyboardInterrupt:
    print("\nProgram interrupted by user.")

finally:
    # Cleanup and shutdown
    # Without torque the arm collapses under its own weight, so wait until it is supported by hand
    try:
        input("Support the arm by hand, then press Enter to disable torque...")
    except (KeyboardInterrupt, EOFError):
        print()
    print("Disabling torque and closing port...")
    for DXL_ID in DXL_IDS:
        packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE)

    portHandler.closePort()
    print("Connection closed.")