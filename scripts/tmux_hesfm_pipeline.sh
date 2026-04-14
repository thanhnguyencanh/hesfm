#!/usr/bin/env bash

set -u

SESSION_NAME="${SESSION_NAME:-hesfm}"
WORKSPACE="${WORKSPACE:-/home/thanhnc19/sslam_ws}"
PACKAGE_NAME="${PACKAGE_NAME:-hesfm}"
CAMERA_NAME="${CAMERA_NAME:-camera}"
SERIAL_NO="${SERIAL_NO:-}"
SEGMENTATION_BACKEND="${SEGMENTATION_BACKEND:-esanet_trt}"
ESANET_DATASET="${ESANET_DATASET:-sunrgbd}"
CONFIG_FILE="${CONFIG_FILE:-$WORKSPACE/src/hesfm/config/hesfm_params.yaml}"
USE_NODELETS="${USE_NODELETS:-true}"
LAUNCH_OV_MSCKF="${LAUNCH_OV_MSCKF:-true}"
LAUNCH_RVIZ="${LAUNCH_RVIZ:-true}"
SEGMENTATION_MONITOR="${SEGMENTATION_MONITOR:-false}"
PUBLISH_PRIMITIVES="${PUBLISH_PRIMITIVES:-false}"
PUBLISH_UNCERTAINTY_MAP="${PUBLISH_UNCERTAINTY_MAP:-false}"
PUBLISH_UNCERTAINTY_CLOUD="${PUBLISH_UNCERTAINTY_CLOUD:-false}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed."
  exit 1
fi

if [ ! -f "$WORKSPACE/devel/setup.bash" ]; then
  echo "Workspace setup file not found: $WORKSPACE/devel/setup.bash"
  exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
  echo "Config file not found: $CONFIG_FILE"
  exit 1
fi

if [ -n "${TMUX:-}" ]; then
  echo "Already inside tmux. Exit this session first, then run the launcher."
  exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Session $SESSION_NAME already exists. Attaching to it."
  exec tmux -2 attach-session -t "$SESSION_NAME"
fi

ROS_ENV="source /opt/ros/noetic/setup.bash && source \"$WORKSPACE/devel/setup.bash\""
CAMERA_CMD="bash -lc '$ROS_ENV && roslaunch $PACKAGE_NAME d455_camera.launch camera_name:=$CAMERA_NAME serial_no:=$SERIAL_NO; status=\$?; echo; echo \"camera pane exited with status \$status\"; exec bash'"
SEGMENTATION_CMD="bash -lc '$ROS_ENV && roslaunch $PACKAGE_NAME segmentation_only.launch launch_camera:=false camera_name:=$CAMERA_NAME backend:=$SEGMENTATION_BACKEND esanet_dataset:=$ESANET_DATASET visualize:=false rviz:=false monitor:=$SEGMENTATION_MONITOR publish_uncertainty:=true; status=\$?; echo; echo \"segmentation pane exited with status \$status\"; exec bash'"
MAPPER_CMD="bash -lc '$ROS_ENV && roslaunch $PACKAGE_NAME mapper_only.launch camera_name:=$CAMERA_NAME esanet_dataset:=$ESANET_DATASET config_file:=$CONFIG_FILE use_nodelets:=$USE_NODELETS launch_ov_msckf:=$LAUNCH_OV_MSCKF publish_primitives:=$PUBLISH_PRIMITIVES publish_uncertainty_map:=$PUBLISH_UNCERTAINTY_MAP publish_uncertainty_cloud:=$PUBLISH_UNCERTAINTY_CLOUD; status=\$?; echo; echo \"mapper pane exited with status \$status\"; exec bash'"
RVIZ_CMD="bash -lc '$ROS_ENV && rviz -d \"$WORKSPACE/src/hesfm/rviz/hesfm.rviz\"; status=\$?; echo; echo \"rviz pane exited with status \$status\"; exec bash'"
IDLE_CMD="bash -lc '$ROS_ENV && echo \"RViz disabled. Pane ready.\" && exec bash'"

tmux new-session -d -s "$SESSION_NAME"
tmux setw -g mouse on

tmux rename-window -t "$SESSION_NAME:0" "Core"
CORE_TOP_PANE="$(tmux display-message -p -t "$SESSION_NAME:Core" '#{pane_id}')"
tmux send-keys -t "$CORE_TOP_PANE" "bash -lc '$ROS_ENV && roscore'" C-m
CORE_BOTTOM_PANE="$(tmux split-window -v -P -F '#{pane_id}' -t "$CORE_TOP_PANE")"
tmux send-keys -t "$CORE_BOTTOM_PANE" "bash -lc '$ROS_ENV && echo \"ROS core window ready.\" && exec bash'" C-m

tmux new-window -t "$SESSION_NAME" -n "Main"
CAMERA_PANE="$(tmux display-message -p -t "$SESSION_NAME:Main" '#{pane_id}')"
SEGMENTATION_PANE="$(tmux split-window -h -P -F '#{pane_id}' -t "$CAMERA_PANE")"
MAPPER_PANE="$(tmux split-window -v -P -F '#{pane_id}' -t "$CAMERA_PANE")"
RVIZ_PANE="$(tmux split-window -v -P -F '#{pane_id}' -t "$SEGMENTATION_PANE")"

tmux send-keys -t "$CAMERA_PANE" "sleep 2; $CAMERA_CMD" C-m
tmux send-keys -t "$SEGMENTATION_PANE" "sleep 4; $SEGMENTATION_CMD" C-m
tmux send-keys -t "$MAPPER_PANE" "sleep 6; $MAPPER_CMD" C-m

if [ "$LAUNCH_RVIZ" = "true" ]; then
  tmux send-keys -t "$RVIZ_PANE" "sleep 8; $RVIZ_CMD" C-m
else
  tmux send-keys -t "$RVIZ_PANE" "$IDLE_CMD" C-m
fi

tmux select-layout -t "$SESSION_NAME:Main" tiled

tmux new-window -t "$SESSION_NAME" -n "Kill"
tmux send-keys -t "$SESSION_NAME:Kill" "tmux kill-session -t $SESSION_NAME"

tmux select-window -t "$SESSION_NAME:Main"
exec tmux -2 attach-session -t "$SESSION_NAME"
