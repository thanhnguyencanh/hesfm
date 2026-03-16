SemanticSlamWs="/home/thanhnc19/sslam_ws/ctz_uav" # point to your workspace directory
SemanticSlamCodeDir="/home/thanhnc19/sslam_ws/ctz_uav/src/" # point to your code directory where you cloned the repository
BAGS_DIR='/home/thanhnc19/activeslam_dataset' # point to your bags / data directory

xhost +local:root # for the lazy and reckless
docker run -it \
    --name="semanticslam_ros" \
    --net="host" \
    --privileged \
    --gpus="all" \
    --workdir="/opt/semanticslam_ws" \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --env="XAUTHORITY=$XAUTH" \
    --volume="$SemanticSlamWs:/opt/semanticslam_ws" \
    --volume="$SemanticSlamCodeDir:$SemanticSlamCodeDir" \
    --volume="$BAGS_DIR:/opt/bags" \
    --volume="/home/$USER/.bash_aliases:/root/.bash_aliases" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="/home/$USER/repos:/home/$USER/repos" \
    activesemanticslam:latest \
    bash