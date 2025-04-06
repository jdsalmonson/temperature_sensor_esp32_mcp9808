cmake_minimum_required(VERSION 3.16)

# from ~/stash/tf-fomo/esp32/fomo_camera_to_servos:
# set(EXTRA_COMPONENT_DIRS components/)
# Taken from my configuration of ~/stash/esp-idf-lib/examples/pca9685/default/CMakeLists.txt
set(EXTRA_COMPONENT_DIRS $ENV{HOME}/stash/esp-idf-lib/components)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(temperature_sensor_mcp9808)
