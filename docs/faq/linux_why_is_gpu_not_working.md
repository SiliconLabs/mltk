# Linux: Why isn't the GPU working?

First, ensure you have correctly followed the Tensorflow [Linux GPU Setup](https://www.tensorflow.org/install/gpu#linux_setup) guide.
If training with the GPU worked at one point, but then stopped working with GPU error messages being printed during the [train](../guides/model_training.md) command,
one possible solution is to reload the GPU driver with the following commands:

```shell
# Stop the CUDA driver
sudo service gdm3 stop

sudo rmmod nvidia_uvm
sudo rmmod nvidia_drm
sudo rmmod nvidia_modeset
sudo rmmod nvidia

# Reload the driver
sudo modprobe nvidia
sudo modprobe nvidia_modeset
sudo modprobe nvidia_drm
sudo modprobe nvidia_uvm
```
