import numpy as np

BASE_WORLD_PATH = "base_worlds/base_world_wide.world" # use wider world
BOTTOM_LEFT = (-5, 3)
TOP_RIGHT = (4, 9.5)
DIRECTIONS = ["R2L", "L2R", "T2B"]

def make_moving_model(index, x_1, y_1, x_2, y_2, direction, time):
  # left to right: 3.14
  # right to left: -3.14
  # top to bottom: -1.57
  # bottom to top: 1.57
    if direction == "R2L":
      rot = 3.14
    elif direction == "L2R":
      rot = -3.14
    elif direction == "T2B":
      rot = -1.57
    elif direction == "B2T":
      rot = 1.57
    return '\n\
    <actor name="actor_%s">\n\
      <plugin name="actor_collisions_plugin" filename="libActorCollisionsPlugin.so">\n\
        <scaling collision="LHipJoint_LeftUpLeg_collision" scale="0.01 0.001 0.001"/>\n\
        <scaling collision="LeftUpLeg_LeftLeg_collision" scale="8.0 8.0 1.0"/>\n\
        <scaling collision="LeftLeg_LeftFoot_collision" scale="8.0 8.0 1.0"/>\n\
        <scaling collision="LeftFoot_LeftToeBase_collision" scale="4.0 4.0 1.5"/>\n\
        <scaling collision="RHipJoint_RightUpLeg_collision" scale="0.01 0.001 0.001"/>\n\
        <scaling collision="RightUpLeg_RightLeg_collision" scale="8.0 8.0 1.0"/>\n\
        <scaling collision="RightLeg_RightFoot_collision" scale="8.0 8.0 1.0"/>\n\
        <scaling collision="RightFoot_RightToeBase_collision" scale="4.0 4.0 1.5"/>\n\
        <scaling collision="LowerBack_Spine_collision" scale="12.0 20.0 5.0" pose="0.05 0 0 0 -0.2 0"/>\n\
        <scaling collision="Spine_Spine1_collision" scale="0.01 0.001 0.001"/>\n\
        <scaling collision="Neck_Neck1_collision" scale="0.01 0.001 0.001"/>\n\
        <scaling collision="Neck1_Head_collision" scale="5.0 5.0 3.0"/>\n\
        <scaling collision="LeftShoulder_LeftArm_collision" scale="0.01 0.001 0.001"/>\n\
        <scaling collision="LeftArm_LeftForeArm_collision" scale="5.0 5.0 1.0"/>\n\
        <scaling collision="LeftForeArm_LeftHand_collision" scale="5.0 5.0 1.0"/>\n\
        <scaling collision="LeftFingerBase_LeftHandIndex1_collision" scale="4.0 4.0 3.0"/>\n\
        <scaling collision="RightShoulder_RightArm_collision" scale="0.01 0.001 0.001"/>\n\
        <scaling collision="RightArm_RightForeArm_collision" scale="5.0 5.0 1.0"/>\n\
        <scaling collision="RightForeArm_RightHand_collision" scale="5.0 5.0 1.0"/>\n\
        <scaling collision="RightFingerBase_RightHandIndex1_collision" scale="4.0 4.0 3.0"/>\n\
      </plugin>\n\
      <skin>\n\
        <filename>walk.dae</filename>\n\
      </skin>\n\
      <animation name="walking">\n\
        <filename>walk.dae</filename>\n\
        <interpolate_x>true</interpolate_x>\n\
      </animation>\n\
      <script>\n\
        <loop>true</loop>\n\
        <auto_start>true</auto_start>\n\
        <trajectory id="0" type="walking">\n\
          <waypoint>\n\
            <time>0</time>\n\
            <pose>%s %s 0 0 0 %s</pose>\n\
          </waypoint>\n\
          <waypoint>\n\
            <time>%s</time>\n\
            <pose>%s %s 0 0 0 %s</pose>\n\
          </waypoint>\n\
        </trajectory>\n\
      </script>\n\
    </actor>\n\
' %(index, x_1, y_1, rot, time, x_2, y_2, rot)

def sample_waypoints(direction, min_speed, max_speed):
    assert direction in DIRECTIONS, "direction %s not defined!" %(direction)
    x1, y1 = BOTTOM_LEFT
    x2, y2 = TOP_RIGHT
    if direction == "B2T":
        start_x = np.random.random() * (x2 - x1) + x1
        start_y = y1
        end_x = np.random.random() * (x2 - x1) + x1
        end_y = y2
    elif direction == "T2B":
        end_x = np.random.random() * (x2 - x1) + x1
        end_y = y1
        start_x = np.random.random() * (x2 - x1) + x1
        start_y = y2
    elif direction == "R2L":
        start_x = x1
        start_y = np.random.random() * (y2 - y1) + y1
        end_x = x2
        end_y = np.random.random() * (y2 - y1) + y1
    elif direction == "L2R":
        end_x = x1
        end_y = np.random.random() * (y2 - y1) + y1
        start_x = x2
        start_y = np.random.random() * (y2 - y1) + y1

    distance = ((start_x - end_x) ** 2 + (start_y - end_y) ** 2) ** 0.5
    speed = np.random.uniform(min_speed, max_speed)
    time = distance / speed
    return direction, [start_x, start_y, 0], [end_x, end_y, time]
    
if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="worlds")
    parser.add_argument('--seed', type=int, default=14)
    parser.add_argument('--min_speed', type=float, default=0.1)
    parser.add_argument('--max_speed', type=float, default=1.5)
    parser.add_argument('--min_object', type=int, default=3)
    parser.add_argument('--max_object', type=int, default=6)
    parser.add_argument('--start_idx', type=int, default=110)
    parser.add_argument('--n_worlds', type=int, default=20)
    parser.add_argument('--rebuild_plugin', action="store_true")
    parser.add_argument('--plugins_per_direction', type=int, default=20)
    args = parser.parse_args()
    np.random.seed(args.seed)
    
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    np.random.seed(args.seed + 1)
    
    with open(BASE_WORLD_PATH, "r") as f:
        ss = f.read()
        part1 = ss.split("TOKEN")[0]
        part2 = ss.split("TOKEN")[1]
        
    for i in range(args.n_worlds):
        mid = ""
        for j in range(np.random.randint(args.min_object, args.max_object)):
            direction, start, end = sample_waypoints(np.random.choice(DIRECTIONS), args.min_speed, args.max_speed)
            mid += make_moving_model(j, start[0], start[1], end[0], end[1], direction, end[2])
            
        with open(os.path.join(args.save_dir, "world_%d.world" %(i + args.start_idx)), "w") as f:
            f.write(part1 + mid + part2)
    print("done")
