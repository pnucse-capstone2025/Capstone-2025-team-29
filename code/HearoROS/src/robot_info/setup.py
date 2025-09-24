from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'robot_info'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models'), glob('models/*')),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_location = robot_info.robot_location:main',
            'camera_info = robot_info.camera_info:main' ,
            'robot_map_location = robot_info.robot_map_location:main',
            'person_detector_node = robot_info.person_detector_node:main',
        ],
    },
)
