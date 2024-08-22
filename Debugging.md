# Reconstruction
- vertical lines in reconstruction
	- excessive learning rate?
# Jwave
- TypeError: Scanned function carry input and carry output must have equal types (e.g. shapes and dtypes of arrays), but they differ:
  * the input carry component fields[0].params has type float32[128,128,1] but the corresponding output carry component has type float32[1,128,128,1], so the shapes do not match
- ValueError: axis 1 is out of bounds for array of dimension 1
	- expand the dims
- AttributeError: BatchTracer has no attribute params
	- Did you discretize?
- TypeError: 'object' object is not subscriptable
	- initialize simulator with sensors object. don't pass sensors object as a parameter
- ValueError: axis 2 is out of bounds for array of dimension 2
	- explicitly define dx in domain 🌟
# Vedo
- TypeError: SetDimensions argument 1: 
	- p_r probably has an extra dimension
- AttributeError: 'NoneType' object has no attribute 'ComputeVisiblePropBounds'
# Flax
- flax.errors.ScopeParamShapeError: Initializer expected to generate shape (1, 1, 1, 1) but got shape (1, 1, 4, 1) instead for parameter "kernel" in "/Conv_1". (https://flax.readthedocs.io/en/latest/api_reference/flax.errors.html#flax.errors.ScopeParamShapeError)
	- Initializing with a different shape than that used in Model.apply()
- flax.errors.ScopeCollectionNotFound: Tried to access "kernel" from collection "params" in "/EncoderBlock_0/Conv_0" but the collection is empty.
	- 
- ValueError: dtype=dtype([('float0', 'V')]) is not a valid dtype for JAX type promotion.
	- [ X ] move FourierSeries discretization outside jitted function
- AttributeError: 'FourierSeries' object has no attribute 'ndim'
	- d_mu and d_c were FourierSeries
- ValueError: operands could not be broadcast together with shapes (0,) (2,) 
	- params initialized without extra dim, or extra dim in wrong place. Images should be (width, heigh, channels), params should be (conv width, conv height, in channels, out channels)
- ValueError: shape must have length equal to the number of dimensions of x;  (2, 64, 32) vs (1, 32, 32, 64)
	- resize getting wrong shape
- flax.errors.ScopeParamShapeError: Initializer expected to generate shape (3, 3, 1, 16) but got shape (3, 3, 128, 16) instead for parameter "kernel" in "/EncoderBlock_0/Conv_0". (https://flax.readthedocs.io/en/latest/api_reference/flax.errors.html#flax.errors.ScopeParamShapeError)
	- params initialized with extra dim ...
# jaxdf
- IndexError: Indexing is only supported if there's at least one batch / time dimension
	- you're trying to index an object shaped (* N, 1) as if it were shaped (1, * N, 1). remove the indexing
# Jax
- TypeError: '<' not supported between instances of 'int' and 'str'
	- This occurs within a jitted function when a number becomes 'nan'  
- TypeError: Cannot interpret value of type <class 'function'> as an abstract array; it does not have a dtype attribute
	- passing a function into a jitted function does not work. use static argnums.
- TypeError: Cannot interpret value of type <class '__main__.RegularizerCNN'> as an abstract array; it does not have a dtype attribute
	- you're jitting with a model in parameters
- TypeError: div got incompatible shapes for broadcasting: (128, 128, 2), (128, 128, 128).
	- use mu_r[0] and c_r[0]
- TypeError: Cannot interpret value of type <class '__main__.RegularizerCNN'> as an abstract array; it does not have a dtype attribute
	- 
- vjp outputting [[[(b'',)]
  [(b'',)]
  [(b'',)]
	- the variable is probably an int instead of a float
- ValueError: unexpected tree structure of argument to vjp function: got PyTreeDef(*), but expected to match PyTreeDef(CustomNode(FourierSeries[('params', 'domain'), (), ()], [*, CustomNode(Domain[(), ('N', 'dx'), ((128, 128), (0.0001, 0.0001))], [])]))
	- vjp input does not match function output
- TypeError: mul got incompatible shapes for broadcasting: (128, 128, 1), (20, 128, 128).
	- Discretize ATT_masks
- AttributeError 'jaxlib.xla_extension.ArrayImpl' object has no attribute 'items'
	- incorrectly initializing parameters
- ValueError: vmap was requested to map its argument along axis 0, which implies that its rank should be at least 1, but is only 0 (its shape is ())
	- Remove keyword arguments from the function and the call. avoid using named arguments with vmap
- ValueError: vmap in_axes must be an int, None, or a tuple of entries corresponding to the positional arguments passed to the function, but got len(in_axes)=5, len(args)=4
	- Remove keyword arguments from the function and the call
- ValueError: Object arrays cannot be loaded when allow_pickle=False
	- files are being saved as discretized versions
- ValueError: Incompatible shapes for broadcasting: shapes=[(256, 256, 128, 3), (256, 256, 128)]
	- Make sure N and dx are tuples 💫
- gpureload not working
	- https://askubuntu.com/questions/1166317/module-nvidia-is-in-use-but-there-are-no-processes-running-on-the-gpu
```python
import os # Set the environment variable os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
```
- jnp.ones((x, x, x)) never initializes for x>64. 
	- Don't use site packages. use a clean venv and install jax with pip
- ValueError: The to_iterable function for a custom PyTree node should return a (children, aux_data) tuple where 'children' is iterable, got (None, (([228, 227, 224, 220, 214, 207, 198, 188, 178, 166, 153, 141, 128, 114, 102, 89, 78, 67, 57, 48, 41, 35, 31, 28, 28, 28, 31, 35, 41, 48, 57, 67, 77, 89, 102, 114, 127, 141, 153, 166, 177, 188, 198, 207, 214, 220, 224, 227], [128, 141, 153, 166, 178, 188, 198, 207, 214, 220, 224, 227, 228, 227, 224, 220, 214, 207, 198, 188, 178, 166, 153, 141, 128, 114, 102, 89, 78, 67, 57, 48, 41, 35, 31, 28, 28, 28, 31, 35, 41, 48, 57, 67, 77, 89, 102, 114]),))
	- data structure is not being properly converted to a JAX array
# Optax
- TypeError: unsupported operand type(s) for *: 'list' and 'ArrayImpl'
	- passing an array of learning rates to an optimizer

# Pickle
- AttributeError: Can't pickle local object 'chain.<locals>.init_fn'
  - 
# Orbax
- ValueError: Comparator raised exception while sorting pytree dictionary keys.
	- Try using PyTreeSave and PyTreeRestore instead of StandardSave and StandardRestore 
# Python script
- UnboundLocalError: cannot access local variable 'exit_flag' where it is not associated with a value

# GPU
## System monitor lost GPU sensors
```shell
sudo modprobe nvidia
sudo modprobe nvidia_uvm
sudo modprobe nvidia_drm
sudo modprobe nvidia_modeset
```
## Jax won't release GPU
```python
jax.clear_caches()
```
Try:
```python
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
```
Reset gpu using `nvidia-smi`: will request termination of processes
```shell
sudo nvidia-smi -r
```
Manually unload and reload Nvidia kernel modules
```shell
sudo rmmod nvidia_uvm
sudo rmmod nvidia_drm
sudo rmmod nvidia_modeset
sudo rmmod nvidia
sudo modprobe nvidia
sudo modprobe nvidia_modeset
sudo modprobe nvidia_drm
sudo modprobe nvidia_uvm

```
List installed GPU drivers
```shell
mhwd -li  
```
List loaded kernel modules
```shell
lsmod | grep nvidia
lmsod | grep i915
```
Show processes currently using Nvidia device
```shell
sudo fuser -v /dev/nvidia*

sudo lsof /dev/nvidia*
```
Optimus-manager
```shell
pamac install optimus-manager
```
Switch between gpu
```shell
optimus-manager --switch integrated

ERROR: no state file found. Is optimus-manager.service running ?
Cannot execute command because of previous errors.

sudo systemctl enable optimus-manager.service 
sudo systemctl start optimus-manager.service 

ERROR: the latest GPU setup attempt failed at Xorg pre-start hook.
Log at /var/log/optimus-manager/switch/switch-20240709T171231.log
Cannot execute command because of previous errors.
```
## GPU tied up by Xorg
⚠️ Executing this command kicks us to another tty, and we can ctrl+alt+F1 to the desktop, where everything appears normal, except that `sudo lsof /dev/nvidia*` show Xorg under new PIDs. 
```shell
sudo killall Xorg
```

Find out your display manager service 
```shell
systemctl list-units | grep -E 'gdm|sddm|lightdm' 
```
Assuming it's sddm, you would stop it like this: 
```shell
sudo systemctl stop sddm
```

```shell
sudo systemctl restart sddm
```
