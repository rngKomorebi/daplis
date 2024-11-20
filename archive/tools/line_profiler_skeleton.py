from line_profiler import LineProfiler as tlp

# from memory_profiler import LineProfiler as mlp


def the_function():
    # do something
    pass


def time_profile_function():
    lp = tlp()
    lp.add_function(the_function)
    lp_wrapper = lp(the_function)
    lp_wrapper()
    lp.print_stats()


# def memory_profile_function():
#     lp = mlp()
#     lp.add_function(calculate_and_save_timestamp_differences_fast)
#     lp_wrapper = lp(calculate_and_save_timestamp_differences_fast)
#     lp_wrapper()
#     lp.print_stats()

time_profile_function()
time_profile_function()
