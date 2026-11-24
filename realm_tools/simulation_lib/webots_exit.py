"""Make Webots exit when its controller does, however the controller ends.

Webots does not stop when its controller dies. A controller that raises
leaves the simulator running, and on a GPU node that means it keeps the CUDA
context the driver gave it for rendering, and the memory with it. The job it
belongs to has already failed and produced nothing; the card stays occupied
until the walltime expires anyway.

How much that costs the next job depends on the card's compute mode, which
is the admin's to set and has been observed as Default on GPU3 -- where a
second context is allowed, so a leftover Webots costs memory and throughput
rather than the whole card. Under Exclusive_Process it costs the whole card.
Neither is a reason to leave one running.

That is how the failure gets out of the job that caused it. The next thing
scheduled onto that node is refused a context it was correctly allocated,
and reports cudaErrorDevicesUnavailable from somewhere entirely unrelated --
a place-field fit, say, two minutes into loading a 2.6 GB dataset. Nothing
in that job's log points back here.

The place-field repo this one was forked from never hit it: its Webots jobs
are collection and render checks, both wrapped in `timeout`, both minutes to
hours. Training is the new shape -- a 24-hour walltime, which is 24 hours of
held card after a crash at minute two.

So every controller registers this, and quits Webots itself:

    quit_webots = quit_on_crash(robot)
    ...
    quit_webots(0)          # the normal end, instead of simulationQuit(0)

A crash, or any exit that never reaches that call, quits with status 1. It
does not catch SIGKILL -- nothing in-process can -- so the `timeout` wrapper
in the job scripts remains the outer backstop.
"""

import atexit
import sys


def quit_on_crash(robot, crash_status=1):
    """Register the exit hooks. Returns the function to end the run with."""
    state = {'quit': False}

    def quit_webots(status=crash_status, why=None):
        if state['quit']:
            return
        state['quit'] = True
        if why:
            print(f'{why}: quitting Webots so it does not sit on the GPU',
                  file=sys.stderr, flush=True)
        try:
            robot.experiment_supervisor.simulationQuit(status)
        except Exception:                                   # noqa: BLE001
            # Already gone, or no supervisor. Nothing useful to do, and
            # raising here would replace the real traceback with this one.
            pass

    def on_exception(exc_type, exc, tb):
        sys.__excepthook__(exc_type, exc, tb)
        quit_webots(crash_status, why='controller crashed')

    sys.excepthook = on_exception
    atexit.register(quit_webots, crash_status,
                    'controller exited without quitting Webots')
    return quit_webots
