cmake_minimum_required(VERSION 2.8)
project(hesfm)


cmake_minimum_required(VERSION 3.0.2)
project(hesfm)

#==============================================================================
# Compiler Settings
#==============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

#==============================================================================
# Find Dependencies
#==============================================================================

# Catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  cv_bridge
  image_transport
  tf2
  tf2_ros
  tf2_sensor_msgs
  tf2_geometry_msgs
  pcl_ros
  pcl_conversions
  message_generation
  dynamic_reconfigure
  nodelet
)

# System packages
find_package(Eigen3 REQUIRED)
find_package(PCL 1.10 REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem)

# Optional: CUDA for GPU acceleration
find_package(CUDA QUIET)
if(CUDA_FOUND)
  message(STATUS "CUDA found: ${CUDA_VERSION}")
  add_definitions(-DUSE_CUDA)
endif()

#==============================================================================
# Messages, Services, Actions
#==============================================================================

# Custom messages
add_message_files(
  FILES
  ExplorationGoal.msg
  ExplorationGoalArray.msg
  GaussianPrimitive.msg
  GaussianPrimitiveArray.msg
  SemanticMap.msg
  SemanticMapCell.msg
  SemanticPoint.msg
  SemanticPointCloud.msg
  UncertaintyInfo.msg
)

# Custom services
add_service_files(
  FILES
  GetExplorationGoals.srv
  GetMapInfo.srv
  GetSemanticMap.srv
  QueryPoint.srv
  QueryRegion.srv
  ResetMap.srv
  SaveMap.srv
  SetParameters.srv
)

# # Dynamic reconfigure
# generate_dynamic_reconfigure_options(
#   cfg/HESFM.cfg
#   cfg/Uncertainty.cfg
#   cfg/Kernel.cfg
# )

# Generate messages
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
)

#==============================================================================
# Catkin Package
#==============================================================================

catkin_package(
  INCLUDE_DIRS 
    include
  LIBRARIES 
    hesfm_core
    hesfm_uncertainty
    hesfm_primitives
    hesfm_kernel
    hesfm_map
  CATKIN_DEPENDS 
    roscpp 
    rospy 
    std_msgs 
    sensor_msgs 
    geometry_msgs 
    nav_msgs
    visualization_msgs 
    cv_bridge 
    image_transport 
    tf2 
    tf2_ros
    tf2_sensor_msgs
    tf2_geometry_msgs 
    pcl_ros 
    pcl_conversions 
    message_runtime
    dynamic_reconfigure
    nodelet
  DEPENDS 
    EIGEN3 
    PCL 
    OpenCV
    Boost
)

#==============================================================================
# Include Directories
#==============================================================================

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

link_directories(
  ${PCL_LIBRARY_DIRS}
)

add_definitions(
  ${PCL_DEFINITIONS}
)

#==============================================================================
# Libraries
#==============================================================================

# # Uncertainty decomposition library
# add_library(hesfm_uncertainty
#   src/hesfm_core/uncertainty.cpp
# )
# target_link_libraries(hesfm_uncertainty
#   ${catkin_LIBRARIES}
#   ${PCL_LIBRARIES}
# )
# add_dependencies(hesfm_uncertainty 
#   ${${PROJECT_NAME}_EXPORTED_TARGETS} 
#   ${catkin_EXPORTED_TARGETS}
# )

# # Gaussian primitives library
# add_library(hesfm_primitives
#   src/hesfm_core/gaussian_primitives.cpp
# )
# target_link_libraries(hesfm_primitives
#   ${catkin_LIBRARIES}
#   ${PCL_LIBRARIES}
# )
# add_dependencies(hesfm_primitives 
#   ${${PROJECT_NAME}_EXPORTED_TARGETS} 
#   ${catkin_EXPORTED_TARGETS}
# )

# # Adaptive kernel library
# add_library(hesfm_kernel
#   src/hesfm_core/adaptive_kernel.cpp
# )
# target_link_libraries(hesfm_kernel
#   ${catkin_LIBRARIES}
# )
# add_dependencies(hesfm_kernel 
#   ${${PROJECT_NAME}_EXPORTED_TARGETS} 
#   ${catkin_EXPORTED_TARGETS}
# )

# # Semantic map library
# add_library(hesfm_map
#   src/hesfm_core/semantic_map.cpp
# )
# target_link_libraries(hesfm_map
#   hesfm_kernel
#   ${catkin_LIBRARIES}
#   ${PCL_LIBRARIES}
# )
# add_dependencies(hesfm_map 
#   ${${PROJECT_NAME}_EXPORTED_TARGETS} 
#   ${catkin_EXPORTED_TARGETS}
# )

# # Combined core library
# add_library(hesfm_core
#   src/hesfm_core/uncertainty.cpp
#   src/hesfm_core/gaussian_primitives.cpp
#   src/hesfm_core/adaptive_kernel.cpp
#   src/hesfm_core/semantic_map.cpp
# )
# target_link_libraries(hesfm_core
#   ${catkin_LIBRARIES}
#   ${PCL_LIBRARIES}
#   ${OpenCV_LIBRARIES}
#   ${Boost_LIBRARIES}
# )
# add_dependencies(hesfm_core 
#   ${${PROJECT_NAME}_EXPORTED_TARGETS} 
#   ${catkin_EXPORTED_TARGETS}
# )

#==============================================================================
# Executables
#==============================================================================

# # HESFM Mapper Node
# add_executable(hesfm_mapper_node 
#   src/nodes/hesfm_mapper_node.cpp
# )
# target_link_libraries(hesfm_mapper_node
#   hesfm_core
#   ${catkin_LIBRARIES}
#   ${PCL_LIBRARIES}
#   ${OpenCV_LIBRARIES}
# )
# add_dependencies(hesfm_mapper_node 
#   ${${PROJECT_NAME}_EXPORTED_TARGETS} 
#   ${catkin_EXPORTED_TARGETS}
# )

# # Semantic Cloud Generator Node
# add_executable(semantic_cloud_node 
#   src/nodes/semantic_cloud_node.cpp
# )
# target_link_libraries(semantic_cloud_node
#   ${catkin_LIBRARIES}
#   ${PCL_LIBRARIES}
#   ${OpenCV_LIBRARIES}
# )
# add_dependencies(semantic_cloud_node 
#   ${${PROJECT_NAME}_EXPORTED_TARGETS} 
#   ${catkin_EXPORTED_TARGETS}
# )

# # Navigation Costmap Node
# add_executable(costmap_node 
#   src/nodes/costmap_node.cpp
# )
# target_link_libraries(costmap_node
#   hesfm_core
#   ${catkin_LIBRARIES}
# )
# add_dependencies(costmap_node 
#   ${${PROJECT_NAME}_EXPORTED_TARGETS} 
#   ${catkin_EXPORTED_TARGETS}
# )

# # Map Visualization Node
# add_executable(visualization_node 
#   src/nodes/visualization_node.cpp
# )
# target_link_libraries(visualization_node
#   hesfm_core
#   ${catkin_LIBRARIES}
#   ${PCL_LIBRARIES}
# )
# add_dependencies(visualization_node 
#   ${${PROJECT_NAME}_EXPORTED_TARGETS} 
#   ${catkin_EXPORTED_TARGETS}
# )

#==============================================================================
# Nodelets
#==============================================================================

# add_library(hesfm_nodelets
#   src/nodelets/hesfm_mapper_nodelet.cpp
#   src/nodelets/semantic_cloud_nodelet.cpp
# )
# target_link_libraries(hesfm_nodelets
#   hesfm_core
#   ${catkin_LIBRARIES}
#   ${PCL_LIBRARIES}
#   ${OpenCV_LIBRARIES}
# )
# add_dependencies(hesfm_nodelets 
#   ${${PROJECT_NAME}_EXPORTED_TARGETS} 
#   ${catkin_EXPORTED_TARGETS}
# )

#==============================================================================
# Install
#==============================================================================

# Install libraries
# install(TARGETS 
#     hesfm_core
#     hesfm_uncertainty
#     hesfm_primitives
#     hesfm_kernel
#     hesfm_map
#     hesfm_nodelets
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )

# # Install executables
# install(TARGETS 
#     hesfm_mapper_node 
#     semantic_cloud_node 
#     costmap_node
#     visualization_node
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

# Install headers
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING 
    PATTERN "*.h"
    PATTERN "*.hpp"
)

# # Install Python scripts
# catkin_install_python(PROGRAMS
#   scripts/semantic_segmentation_node.py
#   scripts/dformer_node.py
#   scripts/esanet_node.py
#   scripts/semantic_cloud_node.py
#   scripts/visualization_node.py
#   scripts/evaluation_node.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

# # Install launch files
# install(DIRECTORY launch/
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
#   FILES_MATCHING PATTERN "*.launch"
# )

# # Install config files
# install(DIRECTORY config/
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
#   FILES_MATCHING 
#     PATTERN "*.yaml"
#     PATTERN "*.rviz"
# )

# # Install RViz files
# install(DIRECTORY rviz/
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/rviz
#   FILES_MATCHING PATTERN "*.rviz"
# )

# # Install nodelet plugin description
# install(FILES nodelet_plugins.xml
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#==============================================================================
# Testing
#==============================================================================

# if(CATKIN_ENABLE_TESTING)
#   find_package(rostest REQUIRED)
  
#   # Unit tests
#   catkin_add_gtest(test_uncertainty test/test_uncertainty.cpp)
#   target_link_libraries(test_uncertainty hesfm_uncertainty ${catkin_LIBRARIES})
  
#   catkin_add_gtest(test_primitives test/test_primitives.cpp)
#   target_link_libraries(test_primitives hesfm_primitives ${catkin_LIBRARIES})
  
#   catkin_add_gtest(test_kernel test/test_kernel.cpp)
#   target_link_libraries(test_kernel hesfm_kernel ${catkin_LIBRARIES})
  
#   catkin_add_gtest(test_map test/test_map.cpp)
#   target_link_libraries(test_map hesfm_map ${catkin_LIBRARIES})
  
#   # Integration tests
#   add_rostest(test/hesfm_integration.test)
  
#   # Launch file tests
#   roslaunch_add_file_check(launch)
# endif()

#==============================================================================
# Documentation
#==============================================================================

find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
  set(DOXYGEN_INPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/docs)
  
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in
    ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    @ONLY
  )
  
  add_custom_target(docs
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM
  )
endif()

#==============================================================================
# Status Messages
#==============================================================================

message(STATUS "")
message(STATUS "HESFM Configuration:")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard:   ${CMAKE_CXX_STANDARD}")
message(STATUS "  Eigen version:  ${EIGEN3_VERSION}")
message(STATUS "  PCL version:    ${PCL_VERSION}")
message(STATUS "  OpenCV version: ${OpenCV_VERSION}")
if(CUDA_FOUND)
  message(STATUS "  CUDA version:   ${CUDA_VERSION}")
else()
  message(STATUS "  CUDA:           Not found")
endif()
message(STATUS "")