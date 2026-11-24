"""A visible GPU is not a usable one, and the difference costs a run.

Job 635169 died with cudaErrorDevicesUnavailable inside
feature_sq_distances, after 51 s of loading a 2.6 GB dataset and building a
7.11 GB feature matrix. Nothing was wrong with the job: the card would not
give it a context, and torch.cuda.is_available() does not notice -- it
counts devices, and the capability, name and memory queries read properties.
None of them creates a CUDA context. The first allocation does, and that is
where it failed.

What made the card refuse was never established. Twenty minutes later it was
idle, 0 MiB used, no processes, compute mode Default -- so the
exclusive-process story told at the time does not survive the evidence, and
the gpu_holders() report exists so the next occurrence arrives with its own.

This path only runs when a card is contended, which is to say never on a
laptop and rarely twice in a row on the cluster, so it is exactly the kind
of code that rots unnoticed. These tests pin it.

    python3 tests/test_device_fallback.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
except ImportError:                                         # noqa: BLE001
    print('torch not installed here; nothing to test')
    sys.exit(0)

from realm_tools.place_cell_lib import rules as R

BUSY = ('CUDA error: CUDA-capable device(s) is/are busy or unavailable\n'
        'Search for `cudaErrorDevicesUnavailable\'')


class FakeCard:
    """A card that answers every query and refuses every allocation.

    Sets REALM_GPU_WAIT=0 unless a test asks otherwise: pick_device waits a
    minute for a busy card by default, which is right on a shared node and
    absurd in a test.
    """

    def __init__(self, allocatable, wait='0', arches=('sm_61',)):
        self.allocatable = allocatable
        self.arches = list(arches)
        self.wait = wait
        self.saved = {}
        self.env = {}

    def __enter__(self):
        for k, v in (('REALM_GPU_WAIT', self.wait),
                     ('REALM_REQUIRE_GPU', os.environ.get('REALM_REQUIRE_GPU', '0'))):
            self.env[k] = os.environ.get(k)
            os.environ[k] = v
        self.saved = {
            'is_available': torch.cuda.is_available,
            'get_device_capability': torch.cuda.get_device_capability,
            'get_device_name': torch.cuda.get_device_name,
            'get_device_properties': torch.cuda.get_device_properties,
            'get_arch_list': torch.cuda.get_arch_list,
            'zeros': torch.zeros,
        }
        real_zeros = self.saved['zeros']
        allocatable = self.allocatable

        class Props:
            total_memory = 11_700_000_000

        def zeros(*a, **k):
            if not str(k.get('device', '')).startswith('cuda'):
                return real_zeros(*a, **k)
            if not allocatable:
                raise RuntimeError(BUSY)
            # Stand in for a card that works. The host running these tests
            # need not have one, so the allocation lands on the CPU; what is
            # under test is which branch pick_device takes, not torch.
            k = {k_: v for k_, v in k.items() if k_ != 'device'}
            return real_zeros(*a, **k)

        torch.cuda.is_available = lambda: True
        torch.cuda.get_device_capability = lambda i=0: (6, 1)
        torch.cuda.get_device_name = lambda i=0: 'FakeCard'
        torch.cuda.get_arch_list = lambda: list(self.arches)
        torch.cuda.get_device_properties = lambda i=0: Props()
        torch.zeros = zeros
        return self

    def __exit__(self, *exc):
        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        torch.cuda.is_available = self.saved['is_available']
        torch.cuda.get_device_capability = self.saved['get_device_capability']
        torch.cuda.get_device_name = self.saved['get_device_name']
        torch.cuda.get_device_properties = self.saved['get_device_properties']
        torch.cuda.get_arch_list = self.saved['get_arch_list']
        torch.zeros = self.saved['zeros']
        return False


def test_a_held_card_falls_back_to_cpu():
    with FakeCard(allocatable=False):
        dev = R.pick_device(use_gpu=True, verbose=False)
    assert str(dev) == 'cpu', f'a busy card returned {dev}, so the run dies later'
    return 'a card that refuses allocation returns cpu'


def test_a_free_card_is_still_used():
    """The fallback must not quietly cost everyone the GPU."""
    with FakeCard(allocatable=True):
        dev = R.pick_device(use_gpu=True, verbose=False)
    assert str(dev).startswith('cuda'), f'a working card returned {dev}'
    return 'a card that allocates is still used'


def test_require_gpu_fails_instead_of_falling_back():
    """When the point of the job is the GPU, a silent CPU run is worse."""
    with FakeCard(allocatable=False):
        os.environ['REALM_REQUIRE_GPU'] = '1'
        try:
            dev = R.pick_device(use_gpu=True, verbose=False)
        except RuntimeError as exc:
            assert 'REALM_REQUIRE_GPU' in str(exc), str(exc)
            return 'REALM_REQUIRE_GPU=1 raises rather than returning cpu'
        raise AssertionError(f'returned {dev} instead of raising')


def test_a_busy_card_is_waited_for():
    """The wait exists for a card about to be released, not for a test."""
    import time
    with FakeCard(allocatable=False, wait='3'):
        t0 = time.time()
        R.pick_device(use_gpu=True, verbose=False)
        waited = time.time() - t0
    assert waited >= 2.5, f'gave up after {waited:.1f}s, expected ~3'
    return f'a busy card is retried for REALM_GPU_WAIT ({waited:.1f}s)'


def test_a_card_the_wheel_cannot_target_is_refused_at_once():
    """The real cause of job 635251, and the one waiting cannot fix.

    The card allocates fine for anything built for it. What it cannot do is
    run kernels that were never compiled for sm_61 -- and no amount of
    retrying changes that, so the wait must not even start.
    """
    import time
    wheel = ['sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']
    with FakeCard(allocatable=True, wait='30', arches=wheel):
        t0 = time.time()
        dev = R.pick_device(use_gpu=True, verbose=False)
        took = time.time() - t0
    assert str(dev) == 'cpu', f'sm_61 against {wheel} returned {dev}'
    assert took < 2, f'waited {took:.1f}s for something waiting cannot fix'
    return 'an arch the wheel lacks falls back immediately, without the wait'


def test_no_gpu_is_honoured():
    with FakeCard(allocatable=True):
        dev = R.pick_device(use_gpu=False, verbose=False)
    assert str(dev) == 'cpu', f'--no-gpu returned {dev}'
    return '--no-gpu still forces cpu'


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    width = max(len(t.__name__) for t in tests)
    failed = 0
    for t in tests:
        try:
            print(f'  PASS  {t.__name__:<{width}}  {t()}')
        except Exception as exc:                            # noqa: BLE001
            failed += 1
            print(f'  FAIL  {t.__name__:<{width}}  '
                  f'{exc.__class__.__name__}: {exc}')
    print(f'\n{len(tests) - failed}/{len(tests)} passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
