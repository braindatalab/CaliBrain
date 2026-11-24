.. meta::
   :description: Run CaliBrain from a container image with Apptainer or Docker, with Jupyter, without installing Python packages. For clusters, servers, and reproducible runs.

Container (Apptainer and Docker)
================================

The container image holds Python, a tested set of the packages CaliBrain needs, the
released CaliBrain package, and Jupyter. It has no display, so the 3D brain plot
(``Visualizer.plot_stc_3d_brain``, which needs the ``viz`` extra) is not available in it.
All other plots work. The example data are not inside the image: keep them in a folder on
your computer and pass that folder as ``data_dir``.

You need Apptainer (on a cluster, usually already installed; ask the administrators) or
Docker on your computer. Neither is part of CaliBrain. If you only want to try CaliBrain
on your own computer, :doc:`pypi_installation` is simpler.

There are three ways to get it:

1. **Pull the published image** (the simplest way, nothing to clone).
2. **Build it yourself** from a definition file, for example on a machine without Docker.
3. **Develop inside it**, with your clone of the repository mounted.

The published image is ``ghcr.io/braindatalab/calibrain:<version>`` (for example
``ghcr.io/braindatalab/calibrain:1.1.1``), and ``latest`` is the newest release. A version
number always gives the same packages, so use one when a result must be reproducible.

1. Pull the published image
---------------------------

With `Apptainer <https://apptainer.org>`_ (formerly Singularity), which needs no root
rights and which most computing clusters provide, convert the image to one file:

.. code-block:: bash

   apptainer pull calibrain.sif docker://ghcr.io/braindatalab/calibrain:1.1.1
   apptainer exec calibrain.sif python -c "import calibrain; print(calibrain.__version__)"

Apptainer makes your home directory and the current directory visible inside the
container, so relative paths work as they do outside. Prepare the example data in
``./data`` (the three steps of :doc:`../datasets/index`):

.. code-block:: bash

   apptainer exec calibrain.sif python -c "from calibrain import fetch_example_data; fetch_example_data(data_dir='data')"
   apptainer exec calibrain.sif python -m calibrain.reduce_fwd --data-dir data
   apptainer exec calibrain.sif python -m calibrain.extract_leadfield --data-dir data

Run your own script the same way: ``apptainer exec calibrain.sif python my_workflow.py``.
If the data are outside your home directory, add ``--bind /path/to/data:/data`` and use
``/data`` as ``data_dir``.

With Docker:

.. code-block:: bash

   docker run --rm -it -v "$PWD/data:/data" ghcr.io/braindatalab/calibrain:1.1.1 \
       python -m calibrain.reduce_fwd --data-dir /data

The image runs as the user ``calibrain`` (user id 1000). If the files in the mounted
folder belong to another user id on your computer, add ``--user "$(id -u):$(id -g)"``
and ``-e HOME=/tmp``.

Jupyter
~~~~~~~

Start a notebook server and open the address it prints. With Apptainer the container uses
the network of the host:

.. code-block:: bash

   apptainer exec calibrain.sif jupyter notebook --ip 0.0.0.0 --no-browser

With Docker, publish the port and mount the folder with your notebooks:

.. code-block:: bash

   docker run --rm -it -p 8888:8888 -v "$PWD:/work" ghcr.io/braindatalab/calibrain:1.1.1 \
       jupyter notebook --ip 0.0.0.0 --no-browser

Every tutorial page has a download link for a Jupyter notebook version.

2. Build the image yourself
---------------------------

The definition file ``container/calibrain.def`` builds the same image with Apptainer,
without Docker. Clone the repository and build from its root. ``VERSION`` is a release
number on PyPI:

.. code-block:: bash

   git clone https://github.com/braindatalab/CaliBrain.git
   cd CaliBrain
   apptainer build --build-arg VERSION=1.1.1 calibrain.sif container/calibrain.def

Building needs root rights or ``--fakeroot`` (ask your cluster administrators). The
Docker equivalent is:

.. code-block:: bash

   docker build -f container/Dockerfile --build-arg VERSION=1.1.1 -t calibrain container

3. Develop inside the container
-------------------------------

To run your own changes of the code, mount your clone and install it in editable mode
inside the container. The changes you make on your computer then take effect at once:

.. code-block:: bash

   git clone https://github.com/braindatalab/CaliBrain.git
   apptainer exec --bind "$PWD/CaliBrain:/app" calibrain.sif \
       sh -c "cd /app && python -m pip install --user --no-deps -e . && python -m pytest tests/"

The editable install goes to your home directory (``--user``), because the image itself
is read-only under Apptainer. With Docker, mount the clone at ``/app`` and start a shell:
``docker run --rm -it -v "$PWD/CaliBrain:/app" ghcr.io/braindatalab/calibrain:1.1.1``,
then run ``python -m pip install --user --no-deps -e /app``. The pytest suite needs
``pytest``, which is not in the image: add it with ``python -m pip install --user pytest``.

What is inside
--------------

``container/Dockerfile`` and ``container/calibrain.def`` install ``calibrain==<version>``
and ``notebook`` from PyPI through the same script (``container/install_calibrain.sh``),
with the package versions in ``container/constraints.txt``. That file lists the versions
the release was tested with, so the same image tag always contains the same packages.
The workflow ``publish-container.yml`` builds, tests (with Docker and Apptainer), and
pushes the image after the package has been published to PyPI. Pull requests that change
the container files run the same tests with a wheel built from the checkout
(``container-check.yml``).
