










domain = Domain(u.N, u.DX)
sound_speed = jnp.ones(u.N) * u.C
medium = Medium(domain=domain, sound_speed=sound_speed, pml_size=u.PML_MARGIN)
time_axis = TimeAxis.from_medium(medium, cfl=u.CFL)




def multi_illumination_1(p0, p_data, sensor_positions, num_iteration=10):
    # load model
    # loss
    # update
    # loop

# loss

# update

# loop
