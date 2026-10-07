"""Hand tracking recorder."""


def record(*args, **kwargs):
    """Run the interactive recorder; see proj.recording.record for options."""
    from .recording import record as run
    return run(*args, **kwargs)


def main():
    record()
