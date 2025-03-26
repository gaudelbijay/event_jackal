import numpy as np

BASE_WORLD_PATH = "base_worlds/base_world_wide.world"

def make_head(name, time_interval):
  return "\n\
#include <gazebo/gazebo.hh>\n\
#include <ignition/math.hh>\n\
#include <gazebo/physics/physics.hh>\n\
#include <gazebo/common/common.hh>\n\
#include <stdio.h>\n\
\n\
namespace gazebo\n\
{\n\
  class %s : public ModelPlugin\n\
  {\n\
    public: void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/)\n\
    {\n\
      // Store the pointer to the model\n\
      this->model = _parent;\n\
\n\
        // create the animation\n\
        gazebo::common::PoseAnimationPtr anim(\n\
              // name the animation '%s',\n\
              // make it last 10 seconds,\n\
              // and set it on a repeat loop\n\
              new gazebo::common::PoseAnimation(\"%s\", %.2f, true));\n\
\n\
        gazebo::common::PoseKeyFrame *key;\n\
" %(name, name, name, time_interval)

def make_tail(name):
  return "\n\
        // set the animation\n\
        _parent->SetAnimation(anim);\n\
    }\n\
\n\
    // Pointer to the model\n\
    private: physics::ModelPtr model;\n\
\n\
    // Pointer to the update event connection\n\
    private: event::ConnectionPtr updateConnection;\n\
  };\n\
\n\
  // Register this plugin with the simulator\n\
  GZ_REGISTER_MODEL_PLUGIN(%s)\n\
}\n\
" %(name)

def make_waypoint(time, x, y):
    return "\n\
        key = anim->CreateKeyFrame(%.2f);\n\
        key->Translation(ignition::math::Vector3d(%.2f, %.2f, 0));\n\
        key->Rotation(ignition::math::Quaterniond(0, 0, 0));\n\
" %(time, x, y)

def make_CMakeLists(name_list):
    s = '\n\
    '
    for n in name_list:
        s += "\n\
add_library(%s SHARED %s.cc)\n\
target_link_libraries(%s ${GAZEBO_LIBRARIES})\n\
" %(n, n, n)
    return s

# take plugin, color, two dimensions in range .08-1
def make_moving_model(plugin, color, shape):
    name = plugin.split(".so")[0]
    return "\n\
    <model name='%s'>\n\
      <static>1</static>\n\
      <pose frame=''>-0.225000 0.075000 0.000000 0.000000 0.000000 0.000000</pose>\n\
      <link name='link'>\n\
        <inertial>\n\
          <mass>1</mass>\n\
          <inertia>\n\
            <ixx>0.145833</ixx>\n\
            <ixy>0</ixy>\n\
            <ixz>0</ixz>\n\
            <iyy>0.145833</iyy>\n\
            <iyz>0</iyz>\n\
            <izz>0.125</izz>\n\
          </inertia>\n\
        </inertial>\n\
        <collision name='collision'>\n\
          <geometry>\n\
            %s\n\
          </geometry>\n\
          <max_contacts>10</max_contacts>\n\
          <surface>\n\
            <contact>\n\
              <ode/>\n\
            </contact>\n\
            <bounce/>\n\
            <friction>\n\
              <torsional>\n\
                <ode/>\n\
              </torsional>\n\
              <ode/>\n\
            </friction>\n\
          </surface>\n\
        </collision>\n\
        <visual name='visual'>\n\
          <geometry>\n\
            %s\n\
          </geometry>\n\
          <material>\n\
            <script>\n\
              <name>Gazebo/%s</name>\n\
              <uri>file://media/materials/scripts/gazebo.material</uri>\n\
            </script>\n\
          </material>\n\
        </visual>\n\
        <self_collide>0</self_collide>\n\
        <kinematic>0</kinematic>\n\
        <gravity>1</gravity>\n\
      </link>\n\
      <plugin name='%s' filename='%s'/>\n\
    </model>\n\
" %(name, shape, shape, color, name, plugin)

def sample_waypoints(direction, min_speed, max_speed, idx):
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
    return "wide_%s_%d_%d" %(direction, int(speed * 100), idx), [(0, start_x, start_y), (time, end_x, end_y)]

def make_random_shape():
    shape = np.random.randint(0, 3)
    if shape == 0: #cylinder
        modelShape = "<cylinder>\n\
                <radius>{}</radius>\n\
                <length>{}</length>\n\
                </cylinder>".format(np.random.uniform(.08, .7), np.random.uniform(.08, .7))
    elif shape == 1: #box
        modelShape = "<box>\n\
                <size>{} {} {}</size>\n\
                </box>".format(np.random.uniform(.08, .7), np.random.uniform(.08, .7), np.random.uniform(.08, .7))
    else: #sphere
        modelShape = "<sphere>\n\
                <radius>{}</radius>\n\
                </sphere>".format(np.random.uniform(.08, .7))
    return modelShape

BOTTOM_LEFT = (-5, 3)
TOP_RIGHT = (4, 9.5)

DIRECTIONS = ["R2L", "L2R", "T2B"]

if __name__ == "__main__":
    import argparse
    import subprocess
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="worlds")
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--min_speed', type=float, default=0.1) #####################
    parser.add_argument('--max_speed', type=float, default=1.5) #####################
    parser.add_argument('--min_object', type=int, default=8) #######################
    parser.add_argument('--max_object', type=int, default=12) #######################
    parser.add_argument('--start_idx', type=int, default=90)
    parser.add_argument('--n_worlds', type=int, default=20)
    parser.add_argument('--rebuild_plugin', action="store_true")
    parser.add_argument('--plugins_per_direction', type=int, default=20) ###########
    args = parser.parse_args()
    np.random.seed(args.seed)
    
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    
    #1 build plugins
    plugins_dir = "plugins"
    '''
    if args.rebuild_plugin or not os.path.exists(plugins_dir):
        np.random.seed(args.seed)
        #os.mkdir(plugins_dir)
        name_list = []
        for d in DIRECTIONS:
            for i in range(args.plugins_per_direction):
                name, waypoints = sample_waypoints(d, args.min_speed, args.max_speed, i)
                name_list.append(name)
                fs = make_head(name, waypoints[-1][0] - waypoints[0][0])
                for wp in waypoints:
                    fs += make_waypoint(*wp)
                fs += make_tail(name)
                with open(os.path.join(plugins_dir, name + ".cc"), "w") as f:
                    f.writelines(fs)
        
        cmake_fs = make_CMakeLists(name_list)
        with open(os.path.join(plugins_dir, "CMakeLists.txt"), "a") as f:
            f.writelines(cmake_fs)
        
        #os.mkdir(os.path.join(plugins_dir, "build"))
    
        wd = os.getcwd()
        #os.chdir(os.path.join(wd, plugins_dir, "build"))
        #subprocess.run(["cmake", ".."])
        #subprocess.call("make")       
        os.chdir(wd)
        print("plugins done")
    '''
    #2 create .world files
    np.random.seed(args.seed + 1)
    plugins_build_dir = os.path.join(plugins_dir, "build")
    plugins = [f for f in os.listdir(plugins_build_dir) if f.endswith(".so") and f.startswith("libwide")]
    with open(BASE_WORLD_PATH, "r") as f:
        ss = f.read()
        part1 = ss.split("TOKEN")[0]
        part2 = ss.split("TOKEN")[1]
        
    colors = ["Grey", "DarkGrey", "White", "Black", "Red", "RedBright",
          "Green", "Blue", "SkyBlue", "Yellow", "DarkYellow",
          "Purple", "Turquoise", "Orange", "Indigo"]
    for i in range(args.n_worlds):
        mid = ""
        for j in range(np.random.randint(args.min_object, args.max_object)):
            plugin = np.random.choice(plugins)
            color = np.random.choice(colors)
            shape = make_random_shape()
            mid += make_moving_model(plugin, color, shape)
            
        with open(os.path.join(args.save_dir, "world_%d.world" %(i + args.start_idx)), "w") as f:
            f.write(part1 + mid + part2)
    print("worlds done")
    
