
#include <gazebo/gazebo.hh>
#include <ignition/math.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <stdio.h>

namespace gazebo
{
  class thin_T2B_118_14 : public ModelPlugin
  {
    public: void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/)
    {
      // Store the pointer to the model
      this->model = _parent;

        // create the animation
        gazebo::common::PoseAnimationPtr anim(
              // name the animation 'thin_T2B_118_14',
              // make it last 10 seconds,
              // and set it on a repeat loop
              new gazebo::common::PoseAnimation("thin_T2B_118_14", 5.71, true));

        gazebo::common::PoseKeyFrame *key;

        key = anim->CreateKeyFrame(0.00);
        key->Translation(ignition::math::Vector3d(-1.49, 9.50, 0));
        key->Rotation(ignition::math::Quaterniond(0, 0, 0));

        key = anim->CreateKeyFrame(5.71);
        key->Translation(ignition::math::Vector3d(-3.45, 3.00, 0));
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
  GZ_REGISTER_MODEL_PLUGIN(thin_T2B_118_14)
}
