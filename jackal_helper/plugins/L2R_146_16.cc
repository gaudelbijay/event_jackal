
#include <gazebo/gazebo.hh>
#include <ignition/math.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <stdio.h>

namespace gazebo
{
  class L2R_146_16 : public ModelPlugin
  {
    public: void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/)
    {
      // Store the pointer to the model
      this->model = _parent;

        // create the animation
        gazebo::common::PoseAnimationPtr anim(
              // name the animation 'L2R_146_16',
              // make it last 10 seconds,
              // and set it on a repeat loop
              new gazebo::common::PoseAnimation("L2R_146_16", 3.48, true));

        gazebo::common::PoseKeyFrame *key;

        key = anim->CreateKeyFrame(0.00);
        key->Translation(ignition::math::Vector3d(0.00, 7.61, 0));
        key->Rotation(ignition::math::Quaterniond(0, 0, 0));

        key = anim->CreateKeyFrame(3.48);
        key->Translation(ignition::math::Vector3d(-4.50, 5.23, 0));
        key->Rotation(ignition::math::Quaterniond(0, 0, 0));

        // set the animation
        _parent->SetAnimation(anim);
    }

    // Pointer to the model
    private: physics::ModelPtr model;

    // Pointer to the update event connection
    private: event::ConnectionPtr updateConnection;
  };

  // Register this plugin with the simulator
  GZ_REGISTER_MODEL_PLUGIN(L2R_146_16)
}
