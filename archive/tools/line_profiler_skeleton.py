from line_profiler import LineProfiler as tlp


def the_function(params):
    # do something
    pass


def time_profile_function():
    lp = tlp()
    lp.add_function(the_function)
    lp_wrapper = lp(lambda: the_function(params))
    lp_wrapper()
    lp.print_stats()


time_profile_function()
