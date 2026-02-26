# quick start


- mamba create -n "yourname" python=3.10.12
	- older versions also work until 3.8
- pip install `magic dmlab wheel`
- pip uninstall numpy
- pip install dm_env
- (fork samplefactory)
- clone samplefactory
- cd samplefactory
- pip install -e .\[dev,mujoco,atari,vizdoom\] -> arguments depend on which environment, we do not need any

when encountering numpy.ndarray error, `pip uninstall numpy` and `pip install -e .` again

patch deepmindlab (transparent reward, custom maps)
```
cd deepmindlab_patch
chmod +x patch_deepmindlab.sh
./patch_deepmindlab.sh
```




## Citation

This project is accepted at ICLR 2026
https://openreview.net/forum?id=li1vfqDzRD
code related to the paper is in the folder [sf_xxl](sf_xxl)

