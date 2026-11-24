#!/bin/sh
# Checks an image. The first argument is the command that runs something inside it:
#   sh container/smoke_test.sh "docker run --rm calibrain:test"
#   sh container/smoke_test.sh "apptainer exec calibrain.sif" 1.1.0
# The optional second argument is the version that calibrain must report.
set -eu

run="$1"
expected="${2:-}"

$run python -c "
import calibrain
print('calibrain', calibrain.__version__)
expected = '$expected'
assert not expected or calibrain.__version__ == expected, (calibrain.__version__, expected)
from calibrain import extract_leadfields, reduce_forward_solutions, fetch_example_data
import matplotlib
assert matplotlib.get_backend().lower() == 'agg', matplotlib.get_backend()
"
$run python -m calibrain.reduce_fwd --help > /dev/null
$run python -m calibrain.extract_leadfield --help > /dev/null
$run jupyter notebook --version
echo "smoke test passed"
