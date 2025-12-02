import subprocess
import sys


def test_lstm_training():

    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "lstm.train_imitation_learning.py" "--config=/configs/ci/ci_tiny.py",
        ]
    )

    assert run.returncode == 0
