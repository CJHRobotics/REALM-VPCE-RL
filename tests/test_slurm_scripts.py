"""Syntax-check the shell scripts AND the Python embedded in them.

`bash -n` validates the shell and says nothing about a heredoc's contents,
because to bash a heredoc is just text. So a Python syntax error inside one
is invisible until the job runs it on the cluster -- which is exactly how a
broken preflight shipped and failed a submission instantly, after the shell
had passed every check I ran on it.

Every heredoc whose tag starts with PY is compiled here. Compiling does not
execute, so this is safe for blocks that import torch or talk to Webots.

    python3 tests/test_slurm_scripts.py
"""

import glob
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLURM = f'{REPO}/slurm'

# <<'TAG' ... TAG, quoted so the shell does no expansion. Only quoted
# heredocs are checked: an unquoted one has $VAR substituted before python
# ever sees it, so its pre-substitution text is not valid Python in general.
HEREDOC = re.compile(
    r"<<\s*'(?P<tag>PY\w*)'\s*\n(?P<body>.*?)^(?P=tag)\s*$",
    re.DOTALL | re.MULTILINE)


def scripts():
    return sorted(glob.glob(f'{SLURM}/*.sh'))


def test_shell_syntax():
    bad = []
    for path in scripts():
        r = subprocess.run(['bash', '-n', path], capture_output=True, text=True)
        if r.returncode != 0:
            bad.append(f'{os.path.basename(path)}: {r.stderr.strip()}')
    assert not bad, 'shell syntax errors:\n  ' + '\n  '.join(bad)
    return f'{len(scripts())} scripts pass bash -n'


def test_embedded_python_compiles():
    checked, bad = 0, []
    for path in scripts():
        src = open(path, encoding='utf-8').read()
        for m in HEREDOC.finditer(src):
            tag, body = m.group('tag'), m.group('body')
            # Line number of the heredoc body within the shell file, so a
            # failure points at the real place rather than at line 1.
            offset = src[:m.start('body')].count('\n')
            checked += 1
            try:
                compile(body, f'{os.path.basename(path)}:{tag}', 'exec')
            except SyntaxError as e:
                bad.append(f'{os.path.basename(path)} heredoc {tag} '
                           f'(file line ~{offset + (e.lineno or 0)}): {e.msg}')
    assert checked, 'no PY heredocs found; has the pattern drifted?'
    assert not bad, 'python syntax errors inside heredocs:\n  ' + \
                    '\n  '.join(bad)
    return f'{checked} embedded python blocks compile'


def test_every_script_is_executable_or_sourced():
    """A job script that is not executable still works via sbatch, but a
    login-node wrapper run as `bash x.sh` is easy to get wrong; flag the
    ones that advertise themselves as runnable."""
    missing = []
    for path in scripts():
        head = open(path, encoding='utf-8').readline()
        if head.startswith('#!') and not os.access(path, os.X_OK):
            missing.append(os.path.basename(path))
    assert not missing, f'shebang but not executable: {missing}'
    return f'{len(scripts())} scripts have consistent exec bits'


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    width = max(len(t.__name__) for t in tests)
    failed = 0
    for t in tests:
        try:
            print(f'  PASS  {t.__name__:<{width}}  {t()}')
        except Exception as exc:                            # noqa: BLE001
            failed += 1
            print(f'  FAIL  {t.__name__:<{width}}  {exc}')
    print(f'\n{len(tests) - failed}/{len(tests)} passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
