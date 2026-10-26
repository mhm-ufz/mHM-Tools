#!/usr/bin/env bash
# Install mhm-tools from this checkout into a conda environment, isolated from
# PYTHONPATH, the user site-packages and pip settings of the calling shell.
# A missing environment is created; an existing one gets a fresh mhm-tools.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_VERSION="3.11"
EXTRAS=""
EDITABLE=false
ENV_DIR=""
# compiled dependencies and the CDO program come from conda-forge, pip adds the
# remaining pure Python dependencies from pyproject.toml
CONDA_PACKAGES=(
    "pip"
    "numpy>=1.17.3,<2.4"
    "scipy"
    "pandas"
    "rasterio"
    "pyproj"
    "geopandas>=1.1.3,<2"
    "netcdf4"
    "h5py"
    "cartopy"
    "pyflwdir"
    "crick"
    "cdo"
)

# Print how to call the script.
print_usage() {
    cat <<EOF
Usage: $(basename "$0") [--editable] [--extras EXTRAS] [--python VERSION] ENV_DIR

Install mhm-tools from ${REPO_DIR} into the conda environment ENV_DIR.
ENV_DIR is created with Python VERSION if it does not exist.

  --editable        install in editable mode instead of a built wheel
  --extras EXTRAS   comma separated extras, e.g. test,doc
  --python VERSION  Python version of a new environment (default ${PYTHON_VERSION})
EOF
}

# Print an error message and stop.
fail() {
    echo "Error: $1" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --editable) EDITABLE=true; shift ;;
        --extras) EXTRAS="${2:?--extras needs a value}"; shift 2 ;;
        --python) PYTHON_VERSION="${2:?--python needs a value}"; shift 2 ;;
        -h|--help) print_usage; exit 0 ;;
        -*) print_usage >&2; fail "unknown option $1" ;;
        *) [[ -z "$ENV_DIR" ]] || fail "only one ENV_DIR expected"; ENV_DIR="$1"; shift ;;
    esac
done
[[ -n "$ENV_DIR" ]] || { print_usage >&2; fail "ENV_DIR is missing"; }

if command -v mamba >/dev/null 2>&1; then
    CONDA_COMMAND="mamba"
elif command -v conda >/dev/null 2>&1; then
    CONDA_COMMAND="conda"
else
    fail "neither mamba nor conda was found"
fi

# nothing from the calling shell may change what is imported or installed
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP
for pip_variable in $(env | sed -n 's/^\(PIP_[A-Za-z0-9_]*\)=.*/\1/p'); do
    unset "$pip_variable"
done
export PYTHONNOUSERSITE=1
export PIP_CONFIG_FILE=/dev/null
export PIP_DISABLE_PIP_VERSION_CHECK=1

if [[ -d "$ENV_DIR/conda-meta" ]]; then
    ENV_DIR="$(cd "$ENV_DIR" && pwd)"
    echo "Updating the conda environment ${ENV_DIR}"
    "$CONDA_COMMAND" install --yes --prefix "$ENV_DIR" \
        --channel conda-forge --override-channels "${CONDA_PACKAGES[@]}"
elif [[ -e "$ENV_DIR" ]]; then
    fail "${ENV_DIR} exists but is no conda environment"
else
    echo "Creating the conda environment ${ENV_DIR} with Python ${PYTHON_VERSION}"
    "$CONDA_COMMAND" create --yes --prefix "$ENV_DIR" \
        --channel conda-forge --override-channels \
        "python=${PYTHON_VERSION}" "${CONDA_PACKAGES[@]}"
    ENV_DIR="$(cd "$ENV_DIR" && pwd)"
fi
PYTHON="${ENV_DIR}/bin/python"

# an old (editable) mhm-tools would keep its stale files and version
"$PYTHON" -I -m pip uninstall --yes mhm_tools >/dev/null 2>&1 || true

EXTRAS_SUFFIX="${EXTRAS:+[${EXTRAS}]}"
if [[ "$EDITABLE" == true ]]; then
    "$PYTHON" -I -m pip install --no-cache-dir --editable "${REPO_DIR}${EXTRAS_SUFFIX}"
    EXPECTED_DIR="${REPO_DIR}/src"
else
    WHEEL_DIR="$(mktemp -d)"
    trap 'rm -rf "$WHEEL_DIR"' EXIT
    # building the wheel writes the current git version into the package
    "$PYTHON" -I -m pip wheel --no-cache-dir --no-deps --wheel-dir "$WHEEL_DIR" "$REPO_DIR"
    WHEEL_FILE="$(ls "$WHEEL_DIR"/mhm_tools-*.whl)"
    "$PYTHON" -I -m pip install --no-cache-dir "${WHEEL_FILE}${EXTRAS_SUFFIX}"
    EXPECTED_DIR="$ENV_DIR"
fi

echo "Checking the installation"
"$PYTHON" -I -m pip check
MODULE_FILE="$("$PYTHON" -I -c 'import mhm_tools; print(mhm_tools.__file__)')"
[[ "$MODULE_FILE" == "$EXPECTED_DIR"/* ]] \
    || fail "mhm_tools is imported from ${MODULE_FILE}, expected below ${EXPECTED_DIR}"
"${ENV_DIR}/bin/mhm-tools" --help >/dev/null
echo "Installed mhm-tools $("$PYTHON" -I -c 'import mhm_tools; print(mhm_tools.__version__)') from ${MODULE_FILE}"
