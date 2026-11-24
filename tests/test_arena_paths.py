"""An arena is found by name, so the names have to stay unique.

Every collect, fit and train command used to carry REALM_ARENA_DIR, and a
command that forgot it died on a path. The directory is now found from the
arena's name -- which works only while no name appears in two of them, and
that is what these tests hold in place.

    python3 tests/test_arena_paths.py
"""

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realm_tools.simulation_lib import arena_paths as A

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def names_by_dir():
    out = {}
    for d in A.ARENA_DIRS:
        out[d] = {os.path.basename(p)[:-4] for p in
                  glob.glob(os.path.join(REPO, A.ENVIRONMENTS, d, '*.xml'))}
    return out


def test_no_name_lives_in_two_directories():
    """The assumption the whole lookup rests on."""
    seen, clash = {}, []
    for d, names in names_by_dir().items():
        for n in names:
            if n in seen:
                clash.append(f'{n} in {seen[n]} and {d}')
            seen[n] = d
    assert not clash, ('arena names must be unique across '
                       f'{", ".join(A.ARENA_DIRS)}: ' + '; '.join(clash))
    return f'{len(seen)} arenas, all uniquely named'


def test_every_arena_resolves_to_its_own_directory():
    os.environ.pop(A.OVERRIDE, None)
    n = 0
    for d, names in names_by_dir().items():
        for name in sorted(names):
            got = A.arena_xml(name, REPO)
            assert os.path.exists(got), f'{name} resolved to a missing {got}'
            assert os.path.basename(os.path.dirname(got)) == d, (
                f'{name} lives in {d} but resolved to {got}')
            n += 1
    assert n, 'no arenas found at all'
    return f'{n} arenas resolve without {A.OVERRIDE}'


def test_an_unknown_name_says_where_it_looked():
    os.environ.pop(A.OVERRIDE, None)
    try:
        A.arena_xml('no_such_arena_12345', REPO)
    except FileNotFoundError as exc:
        assert all(d in str(exc) for d in A.ARENA_DIRS), str(exc)
        return 'unknown names name the directories searched'
    raise AssertionError('an unknown arena resolved to something')


def test_the_override_still_forces_a_directory():
    os.environ[A.OVERRIDE] = 'vpce'
    try:
        # vpce-rl's arena is not in vpce, so forcing vpce must fail rather
        # than quietly fall back to the search.
        rl = sorted(names_by_dir()['vpce-rl'])
        if rl:
            try:
                A.arena_xml(rl[0], REPO)
            except FileNotFoundError as exc:
                assert A.OVERRIDE in str(exc)
            else:
                raise AssertionError(f'{A.OVERRIDE}=vpce still found {rl[0]}')
        one = sorted(names_by_dir()['vpce'])[0]
        got = A.arena_xml(one, REPO)
        assert os.path.basename(os.path.dirname(got)) == 'vpce'
    finally:
        os.environ.pop(A.OVERRIDE, None)
    return f'{A.OVERRIDE} overrides the search and fails loudly'


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
