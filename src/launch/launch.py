

import launch
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory

import os

bundle_adjustment_pkg_prefix = get_package_share_directory('bundle_adjustment')
bundle_adjustment_param_file = os.path.join(bundle_adjustment_pkg_prefix,
                                                  'param/params.yaml')


def generate_launch_description():
    bundle_adjustment_node = Node(
        package='bundle_adjustment',
        executable='bundle_adjustment_node_exe',
        namespace='drivers',
        parameters=[bundle_adjustment_param_file],
        #prefix=['valgrind --tool=callgrind --dump-instr=yes -v --instr-atstart=yes'],
        output='screen'
        
    )

    return launch.LaunchDescription([bundle_adjustment_node])
