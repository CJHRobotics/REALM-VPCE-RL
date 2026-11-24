"""Find an arena by name, wherever its directory happens to be.

Arenas live in more than one directory under `simulation/worlds/environments`
and the split is deliberate: `vpce` holds the place-field paper's sweep of
empty enclosures, whose datasets that paper rests on, and `vpce-rl` holds the
obstacle arenas the RL work uses. An arena from one must never be mistaken
for an arena from the other.

Keeping them apart does not require the caller to say which one it means.
Arena NAMES are unique across the directories -- `circ_lm8_r10` and
`circ_lm10_r10_o20` are different arenas and would be whatever directory they
sat in -- so the name alone identifies the file, and every entry point that
takes an arena name can simply look for it. That removes an environment
variable from every collect, fit and train command, and with it the failure
where the variable is forgotten and the job dies on a path.

A name in two directories at once is the one case the name does not settle,
and it is reported rather than guessed: whichever one got picked, half the
time it would be the wrong dataset, and nothing downstream could tell.

    from realm_tools.simulation_lib.arena_paths import arena_xml
    xml = arena_xml('circ_lm10_r10_o20')
"""

import os

ENVIRONMENTS = 'simulation/worlds/environments'

# The directories holding arenas that something runs against. `archive` and
# `samples` are deliberately absent: archive exists to hold superseded copies,
# so searching it would turn any name it shares with a live arena into an
# ambiguity error, and the archived one is never what was meant. Add a
# directory here when it starts holding arenas in use.
ARENA_DIRS = ('vpce', 'vpce-rl')

# Set this to a directory name to skip the search and demand that one.
OVERRIDE = 'REALM_ARENA_DIR'


def arena_dir(name, repo='.'):
    """Directory holding <name>.xml, as a path under `repo`."""
    forced = os.environ.get(OVERRIDE, '').strip().strip('/')
    if forced:
        d = os.path.normpath(os.path.join(repo, ENVIRONMENTS, forced))
        if not os.path.exists(os.path.join(d, name + '.xml')):
            raise FileNotFoundError(
                f'{OVERRIDE}={forced} but there is no {name}.xml in {d}. '
                f'Unset {OVERRIDE} to search '
                f'{", ".join(ARENA_DIRS)} instead.')
        return d

    found = [d for d in ARENA_DIRS
             if os.path.exists(os.path.join(repo, ENVIRONMENTS, d,
                                            name + '.xml'))]
    if len(found) == 1:
        return os.path.normpath(os.path.join(repo, ENVIRONMENTS, found[0]))
    if not found:
        raise FileNotFoundError(
            f'no arena named {name!r} in '
            + ' or '.join(f'{ENVIRONMENTS}/{d}' for d in ARENA_DIRS)
            + '. Check the spelling, or generate it first.')
    raise ValueError(
        f'{name!r} exists in {" and ".join(found)}, so the name does not say '
        f'which arena is meant. Rename one, or set {OVERRIDE} to the '
        f'directory you want.')


def arena_xml(name, repo='.'):
    """Full path to the arena's XML."""
    return os.path.join(arena_dir(name, repo), name + '.xml')


def arena_positions(name, repo='.'):
    """Full path to the arena's collection grid."""
    return os.path.join(arena_dir(name, repo), 'positions',
                        name + '_positions.csv')
