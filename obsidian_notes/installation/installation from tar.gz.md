python=3.10.12
###### Try 1
- create venv
- pip install dmlab wheel
- pip install dm_env
- (fork samplefactory)
- clone samplefactory
- cd samplefactory
- pip install -e .\[dev,mujoco,atari,vizdoom\] -> arguments depend on which environment, we do not need any
- sudo apt-get install (libosmesa6 #notnecessarymaybe) libosmesa6-dev
- clone hipposlam_sf as submodule (git add submodule https://...)
###### Try 2
- exported requirements.txt from venv
- reinstall venv into sample-factory folder in order for vs code to recognize it
###### Try 3 on NEMO
- Use mamba instead of venv
- basically the same but after installing the dmlab wheel:
	- pip uninstall numpy, this is needed to not confuse python with different numpy versions
- continue with the installation
- instead of apt-get libosmesa6-dev, do module load lib/sdl2/2.28.2-gcccore-12.3.0 vis/mesa/24.1.3-gcccore-13.3.0
	- Probably needs installing beforehand but I don't remember the command and can't find it