import numpy as np

def quadspace(start, stop, num, power=0.5, increasing=True, endpoint=True):
    """
    Nonlinear spacing between start and stop.
    power > 1  → more curvature
    power = 0  → uniform (linspace)
    0 < power < 1  → gentler than quadratic
    """

    steps = np.arange(1, num, dtype=float) ** power
    if not increasing:
        steps = steps[::-1]
    steps = steps / steps.sum()

    x = np.concatenate(([0], np.cumsum(steps)))

    if not endpoint:
        x = x[:-1]
    result = start + (stop - start) * x
    if endpoint:
        result[-1] = stop  # Ensure exact stop value
    return result


def compute_timesteps(
        planning_time_to_td,
        duration_stop,
        duration_throw,
        nb_discretization_points,
        k_catch,
        k_throw,
        coll_buffer_time,
        nb_colls_ks,
        throw_step_dt,
    ):
        total_move_time = planning_time_to_td + duration_throw + duration_stop

        if nb_colls_ks == 0:
            coll_buffer_time = 0.0  # No collinearity buffer if no colinearity points
        
        k_colls = 2 * nb_colls_ks + 1
        coll_start = planning_time_to_td - coll_buffer_time
        coll_end = planning_time_to_td + coll_buffer_time
        a = np.linspace(0, coll_start, k_catch - nb_colls_ks, endpoint=False)
        b = np.linspace(coll_start, coll_end, k_colls, endpoint=True)[:-1]  
        c = quadspace(
            planning_time_to_td + duration_throw - throw_step_dt,
            coll_end,
            k_throw - k_catch - nb_colls_ks + 1,
            increasing=True,
            endpoint=True
        )[::-1]
        d = quadspace(
            total_move_time,
            planning_time_to_td + duration_throw + throw_step_dt,
            nb_discretization_points - (k_throw + 1),
            increasing=False,
            endpoint=True
        )[::-1]

        timesteps = np.concatenate([a, b, c, d])
        return a, b, c, d, timesteps

def compute_timesteps_equidistant(
    planning_time_to_td,
    duration_stop,
    duration_throw,
    nb_discretization_points,
    k_catch,
    k_throw,
    coll_buffer_time,
    nb_colls_ks,
    throw_step_dt,
):
    total_move_time = planning_time_to_td + duration_throw + duration_stop

    if nb_colls_ks == 0:
        coll_buffer_time = 0.0  # No collinearity buffer if no colinearity points
    
    k_colls = 2 * nb_colls_ks + 1
    coll_start = planning_time_to_td - coll_buffer_time
    coll_end = planning_time_to_td + coll_buffer_time
    a = np.linspace(0, coll_start, k_catch - nb_colls_ks, endpoint=False)
    b = np.linspace(coll_start, coll_end, k_colls, endpoint=True)[:-1]  
    c = np.linspace(
        planning_time_to_td + duration_throw - throw_step_dt,
        coll_end,
        k_throw - k_catch - nb_colls_ks + 1,
        endpoint=True
    )[::-1]
    d = np.linspace(
        total_move_time,
        planning_time_to_td + duration_throw + throw_step_dt,
        nb_discretization_points - (k_throw + 1),
        endpoint=True
    )[::-1]

    timesteps = np.concatenate([a, b, c, d])
    return a, b, c, d, timesteps

def compute_timesteps_on_k(
        planning_time_to_td,
        duration_stop,
        duration_throw,
        nb_discretization_points,
        k_catch,
        k_throw,
        coll_buffer_time,
        nb_colls_ks,
        throw_step_dt,
    ):
        total_move_time = planning_time_to_td + duration_throw + duration_stop

        if nb_colls_ks == 0:
            coll_buffer_time = 0.0  # No collinearity buffer if no colinearity points
        
        k_colls = 2 * nb_colls_ks + 1
        coll_start = planning_time_to_td - coll_buffer_time
        coll_end = planning_time_to_td + coll_buffer_time
        a = np.linspace(0, coll_start, k_catch - nb_colls_ks, endpoint=False)
        b = np.linspace(coll_start, coll_end, k_colls, endpoint=True)[:-1]  
        c = quadspace(
            planning_time_to_td + duration_throw,
            coll_end,
            k_throw - k_catch - nb_colls_ks + 1,
            increasing=True,
            endpoint=True
        )[::-1]
        d = quadspace(
            total_move_time,
            planning_time_to_td + duration_throw,
            nb_discretization_points - (k_throw ),
            increasing=False,
            endpoint=False
        )[::-1]

        timesteps = np.concatenate([a, b, c, d])
        return a, b, c, d, timesteps
    
    
    
def compute_timesteps_start_catch_beginn_of_throw(
        planning_time_to_td,
        duration_throw,
        k_catch,
        k_throw,
        coll_buffer_time,
        nb_colls_ks,
        throw_step_dt,
    ):
        planning_time_to_td + duration_throw 

        if nb_colls_ks == 0:
            coll_buffer_time = 0.0  # No collinearity buffer if no colinearity points
        
        k_colls = 2 * nb_colls_ks + 1
        coll_start = planning_time_to_td - coll_buffer_time
        coll_end = planning_time_to_td + coll_buffer_time
        a = np.linspace(0, coll_start, k_catch - nb_colls_ks, endpoint=False)
        b = np.linspace(coll_start, coll_end, k_colls, endpoint=True)[:-1]  
        c = quadspace(
            planning_time_to_td + duration_throw,
            coll_end,
            k_throw - k_catch - nb_colls_ks + 1,
            increasing=True,
            endpoint=True
        )[::-1]

        timesteps = np.concatenate([a, b, c])
        return timesteps, (a, b, c,)
    
def compute_timesteps_begin_of_throw_to_stop(
        planning_time_to_td,
        duration_throw,
        duration_stop,
        nb_discretization_points,
        k_throw,
    ):
        total_move_time = planning_time_to_td + duration_throw + duration_stop
        d = quadspace(
            total_move_time,
            planning_time_to_td + duration_throw,
            nb_discretization_points - (k_throw ),
            increasing=False,
            endpoint=True
        )[::-1]

        timesteps = np.concatenate([d])
        return timesteps, (d)
    
def compute_timesteps_begin_of_throw_to_stop_from_zero(
        duration_stop,
        nb_discretization_points,
        k_throw,
    ):
        d = quadspace(
            duration_stop,
            0,
            nb_discretization_points - (k_throw ),
            increasing=False,
            endpoint=True
        )[::-1]

        timesteps = np.concatenate([d])
        return timesteps, (d)



def compute_timesteps_throw_and_stop(
        duration_throw,
        duration_stop,
        nb_discretization_points,
        k_throw,
    ):
        total_move_time = duration_throw + duration_stop
        a = quadspace(
            0,
            duration_throw,
            k_throw + 1,
            increasing=False,
            endpoint=True
        )
        b = quadspace(
            total_move_time,
            duration_throw,
            nb_discretization_points - (k_throw ),
            increasing=False,
            endpoint=False
        )[::-1]

        timesteps = np.concatenate([a, b])
        return timesteps, (a, b)

def _test_compute_timesteps_throw_and_stop():
    duration_throw = 0.3
    duration_stop = 0.3
    nb_discretization_points = 20
    k_throw = 9

    total_move_time = duration_throw + duration_stop

    timesteps, (a, b) = compute_timesteps_throw_and_stop(
        duration_throw,
        duration_stop,
        nb_discretization_points,
        k_throw,
    )
    assert len(timesteps) == nb_discretization_points, "Length of timesteps does not match nb_discretization_points"
    assert timesteps[k_throw] == duration_throw, "k_throw timestep incorrect"
    assert timesteps[0] == 0.0, "First timestep should be 0.0"
    assert timesteps[-1] == total_move_time, "Last timestep should be total move time"
    print("Timesteps:", timesteps)
    

if __name__ == "__main__":
    # _test_compute_timesteps_throw_and_stop()
    # exit()

    # Example usage
    planning_time_to_td = 0.5
    duration_stop = 0.5
    duration_throw = 0.5
    nb_discretization_points = 30
    k_catch = 10
    k_throw = 20
    coll_buffer_time = 0.05
    nb_colls_ks = 0
    throw_step_dt = np.inf

    # a, b, c, d, timesteps = compute_timesteps_on_k(
    #     planning_time_to_td=planning_time_to_td,
    #     duration_stop=duration_stop,
    #     duration_throw=duration_throw,
    #     nb_discretization_points=nb_discretization_points,
    #     k_catch=k_catch,
    #     k_throw=k_throw,
    #     coll_buffer_time=coll_buffer_time,
    #     nb_colls_ks=nb_colls_ks,
    #     throw_step_dt=throw_step_dt,
    # )
    # print("Timesteps:", timesteps)
    # print(timesteps[10], timesteps[20])
    # print(len(timesteps))
    
    # assert len(timesteps) == nb_discretization_points, "Length of timesteps does not match nb_discretization_points"
    # assert timesteps[k_catch] == planning_time_to_td, "k_catch timestep incorrect"
    # assert timesteps[k_throw] == planning_time_to_td + duration_throw, "k_throw timestep incorrect"
    # assert timesteps[0] == 0.0, "First timestep should be 0.0"
    # assert timesteps[-1] == planning_time_to_td + duration_throw + duration_stop, "Last timestep should be total move time"
    
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 4))
    # for t in timesteps:
    #     plt.axvline(x=t, color='gray', linestyle='--', alpha=0.3)
    # plt.axvline(x=timesteps[k_catch], color='red', linestyle='-', label='k_catch')
    # plt.axvline(x=timesteps[k_throw], color='blue', linestyle='-', label='k_throw')
    # plt.legend()
    # plt.title("Timesteps Distribution")
    # plt.xlabel("Time")
    # plt.ylabel("Index")
    # plt.grid(True)
    # plt.show()

    timesteps,_ = compute_timesteps_start_catch_beginn_of_throw(
        planning_time_to_td,
        duration_throw,
        k_catch,
        k_throw,
        coll_buffer_time,
        nb_colls_ks,
        throw_step_dt,
    )
    
    timesteps2,_ = compute_timesteps_begin_of_throw_to_stop(
        planning_time_to_td,
        duration_throw,
        duration_stop,
        nb_discretization_points,
        k_throw,
    )
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    for t in timesteps:
        plt.axvline(x=t, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(x=timesteps[k_catch], color='red', linestyle='-', label='k_catch')
    plt.axvline(x=timesteps[k_throw], color='blue', linestyle='-', label='k_throw')
    plt.legend()
    plt.title("Timesteps Distribution")
    plt.xlabel("Time")
    plt.ylabel("Index")
    plt.grid(True)
    plt.show()
    
    print("Timesteps:", timesteps)
    print(timesteps[10], timesteps[20])
    print(len(timesteps))
    print("Timesteps2:", timesteps2)
    print(len(timesteps2))