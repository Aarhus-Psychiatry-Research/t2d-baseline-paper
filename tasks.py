from pathlib import Path
from typing import Optional

from attr import dataclass
from invoke import Context, Result, task


def echo_header(msg: str):
    print(f"\n--- {msg} ---")


@dataclass
class Emo:
    APPLY = "🤖"
    SUCCESS = "✅"
    FAILURE = "🚨"
    WARNING = "🚧"
    SYNC = "🚂"
    STARTING = "🔨"
    PYTHON = "🐍"
    CLEAN = "🧹"
    TEST = "🧪"
    COMMUNICATE = "📣"


@task
def setup(c: Context, python_version: str = "3.9"):
    git_init(c)
    setup_venv(c, python_version=python_version)
    install(c)


def git_init(c: Context):
    # If no .git directory exits
    if not Path(".git").exists():
        echo_header(f"{Emo.STARTING} Initializing Git repository")
        c.run("git init")
        c.run("git add .")
        c.run("git commit -m 'Initial commit'")
        print(f"{Emo.SUCCESS} Git repository initialized")
    else:
        print(f"{Emo.SUCCESS} Git repository already initialized")


def setup_venv(
    c: Context,
    python_version: str,
):
    venv_name = f'.venv{python_version.replace(".", "")}'

    if not Path(venv_name).exists():
        echo_header(f"{Emo.STARTING} Creating virtual environment")
        c.run(f"python{python_version} -m venv {venv_name}")
        print(f"{Emo.SUCCESS} Virtual environment created")
    else:
        print(f"{Emo.SUCCESS} Virtual environment already exists")

    c.run(f"source {venv_name}/bin/activate")


@task
def install(c: Context):
    echo_header(f"{Emo.STARTING} Installing project")
    c.run("pip install -e '.[dev,tests]'")


@task
def update(c: Context):
    echo_header(f"{Emo.STARTING} Updating project")
    c.run("pip install --upgrade -e '.[dev,tests]'")


@task
def test(c: Context, min_latency: bool = True):
    """Run tests.

    Args:
        min_latency (bool): If false, go for maximum throughput. If true, go for minimum latency.
    """
    echo_header(f"{Emo.TEST} Running tests")

    if min_latency:
        n = ""
        plugins = "-p no:cov -p no:xdist -p no:instafail -p no:xdist"
    else:
        n = "-n auto "

    test_result: Result = c.run(
        f"pytest {n}-rfE --failed-first {plugins} --disable-warnings -q",
        warn=True,
        pty=True,
    )

    # If "failed" in the pytest results
    if "failed" in test_result.stdout:
        print("\n\n\n")
        echo_header("Failed tests")

        # Get lines with "FAILED" in them from the .pytest_results file
        failed_tests = [
            line
            for line in Path("tests/.pytest_results").read_text().splitlines()
            if line.startswith("FAILED")
        ]

        for line in failed_tests:
            # Remove from start of line until /test_
            line_sans_prefix = line[line.find("test_") :]

            # Keep only that after ::
            line_sans_suffix = line_sans_prefix[line_sans_prefix.find("::") + 2 :]
            print(f"FAILED {Emo.FAILURE} #{line_sans_suffix}     ")

    if "failed" in test_result.stdout or "error" in test_result.stdout:
        exit(0)


def _add_commit(c: Context, msg: Optional[str] = None):
    print("🔨 Adding and committing changes")
    c.run("git add .")

    if msg is None:
        msg = input("Commit message: ")

    c.run(f'git commit -m "{msg}"', pty=True, hide=True)
    print("\n🤖 Changes added and committed\n")


def add_and_commit(c: Context, msg: Optional[str] = None):
    """Add and commit all changes."""
    if is_uncommitted_changes(c):
        uncommitted_changes_descr = c.run(
            "git status --porcelain",
            pty=True,
            hide=True,
        ).stdout

        echo_header(
            f"{Emo.WARNING} Uncommitted changes detected",
        )

        input("Press enter to add and commit the changes...")

        for line in uncommitted_changes_descr.splitlines():
            print(f"    {line.strip()}")
        print("\n")
        _add_commit(c, msg=msg)


def is_uncommitted_changes(c: Context) -> bool:
    git_status_result: Result = c.run(
        "git status --porcelain",
        pty=True,
        hide=True,
    )

    uncommitted_changes = git_status_result.stdout != ""
    return uncommitted_changes


@task
def pr(c: Context):
    add_and_commit(c)
    lint(c)
    test(c, min_latency=False)
    sync_with_git_remote(c)
    sync_pr(c)


def sync_with_git_remote(c: Context):
    echo_header(f"{Emo.SYNC} Syncing branch with remote")

    if not branch_exists_on_remote(c):
        c.run("git push --set-upstream origin HEAD")
    else:
        print("Pulling")
        c.run("git pull")
        print("Pushing")
        c.run("git push")


def sync_pr(c: Context):
    echo_header(f"{Emo.COMMUNICATE} Syncing PR")
    # Get current branch name
    branch_name = Path(".git/HEAD").read_text().split("/")[-1].strip()
    pr_result: Result = c.run(
        "gh pr list --state OPEN",
        pty=False,
        hide=True,
    )

    if branch_name not in pr_result.stdout:
        create_pr(c)
    else:
        open_web = input("Open in browser? [y/n] ")
        if "y" in open_web.lower():
            c.run("gh pr view --web", pty=True)


def create_pr(c: Context):
    c.run(
        "gh pr create --web",
        pty=True,
    )


def branch_exists_on_remote(c: Context) -> bool:
    branch_name = Path(".git/HEAD").read_text().split("/")[-1].strip()

    branch_exists_result: Result = c.run(
        f"git ls-remote --heads origin {branch_name}",
        hide=True,
    )

    return branch_name in branch_exists_result.stdout


@task
def lint(c: Context):
    if is_uncommitted_changes(c):
        print(
            f"{Emo.WARNING} Your git working directory is not clean. Stash or commit before linting.",
        )
        exit(0)

    pre_commit(c)
    mypy(c)


def pre_commit(c: Context):
    echo_header(f"{Emo.CLEAN} Running pre-commit checks")
    pre_commit_cmd = "pre-commit run --all-files"
    result = c.run(pre_commit_cmd, pty=True, warn=True)

    exit_if_error_in_stdout(result)

    if "fixed" in result.stdout or "reformatted" in result.stdout:
        _add_commit(c, msg="style: linting")

        print(f"{Emo.APPLY} Fixed errors, re-running pre-commit checks")
        second_result = c.run(pre_commit_cmd, pty=True, warn=True)
        exit_if_error_in_stdout(second_result)


def exit_if_error_in_stdout(result: Result):
    if "error" in result.stdout:
        exit(0)


def mypy(c: Context):
    echo_header(f"{Emo.CLEAN} Running mypy")
    c.run("mypy .", pty=True)