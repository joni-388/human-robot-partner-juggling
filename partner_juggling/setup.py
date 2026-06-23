from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    # scripts=[''],
    packages=['partner_juggling', 'partner_juggling.tracker', 'partner_juggling.online_test',],
    package_dir={'': 'src',}
)

setup(**d)
