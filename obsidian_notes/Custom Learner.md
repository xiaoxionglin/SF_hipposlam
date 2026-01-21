# General
It adapts a similar structure to the injecting of custom model structures, so you need a similar *create_learner_function()* and to register this at the same time point in the *train_hipposlam.py* file. 
###### enjoy remarks
During enjoy this is not necessary, the enjoy function only uses *get_checkpoints(), checkpoint_dir(), load_checkpoint()* from the Learner class as static methods. So when these are changed in a custom Learner, remember to create a custom enjoy as well.
###### separate forward pass
refactored the forward pass to be called multiple times from different locations with some additional features, mainly the ability to record & return the different outputs. Can also be only a forward pass of the encoder only to save some compute.
###### Hooks
I also added the ability to inject custom hooks (both forward and backward) during initialization of the Learner class. This uses the *_register_forward_hooks()/register_forward_hooks()* functions in BaseLearner, which can be overwritten. Technically it would be better to have these functions in the *custom_actor_critic.py*, but I didn't use it in the end, so never properly changed it. It works though.
# Existing additional classes
### BaseDistanceRecorder
Extends *BaseLearner* to implement the recording of the distance metrics even without training on them. Uses the basically the normal 