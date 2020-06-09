#script to run when the system starts up to check fresh image and run container(s)
export LOGFILE="/home/ubuntu/truck_detector/log.txt"
export YAMLFILE="/home/ubuntu/truck_detector/docker-compose.yaml"

echo \n >> $LOGFILE
date >> $LOGFILE
echo "Checking updates.." >> $LOGFILE
docker-compose -f $YAMLFILE  pull  >> $LOGFILE 2>&1
echo "Starting containers.." >> $LOGFILE
docker-compose -f $YAMLFILE up
