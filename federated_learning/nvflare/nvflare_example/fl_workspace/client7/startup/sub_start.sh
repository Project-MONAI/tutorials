#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# export CUDA_VISIBLE_DEVICES=
echo "WORKSPACE set to $DIR/.."
mkdir -p $DIR/../transfer
export PYTHONPATH=/local/custom:$PYTHONPATH
echo "PYTHONPATH is $PYTHONPATH"

SECONDS=0
lst=-400
restart_count=0
start_fl() {
  if [[ $(( $SECONDS - $lst )) -lt 300 ]]; then
    ((restart_count++))
  else
    restart_count=0
  fi
  if [[ $(($SECONDS - $lst )) -lt 300 && $restart_count -ge 5 ]]; then
    echo "System is in trouble and unable to start the task!!!!!"
    rm -f $DIR/../pid.fl $DIR/../shutdown.fl $DIR/../restart.fl $DIR/../daemon_pid.fl
    exit
  fi
  lst=$SECONDS
((python3 -u -m nvflare.private.fed.app.client.client_train -m $DIR/.. -s fed_client.json --set secure_train=true multi_gpu=false uid=client7 config_folder=config 2>&1 & echo $! >&3 ) 3>$DIR/../pid.fl | tee -a $DIR/../log.txt &)
  pid=`cat $DIR/../pid.fl`
  echo "new pid ${pid}"
}

stop_fl() {
  if [[ ! -f "$DIR/../pid.fl" ]]; then
    echo "No pid.fl.  No need to kill process."
    return
  fi
  pid=`cat $DIR/../pid.fl`
  sleep 15
  kill -0 ${pid} 2> /dev/null 1>&2
  if [[ $? -ne 0 ]]; then
    echo "Process already terminated"
    return
  fi
  rkill -9 $pid
  rm -f $DIR/../pid.fl $DIR/../shutdown.fl $DIR/../restart.fl
}
  
if [[ -f "$DIR/../daemon_pid.fl" ]]; then
  dpid=`cat $DIR/../daemon_pid.fl`
  kill -0 ${dpid} 2> /dev/null 1>&2
  if [[ $? -eq 0 ]]; then
    echo "There seems to be one instance, pid=$dpid, running."
    echo "If you are sure it's not the case, please kill process $dpid."
    exit
  fi
  rm -f $DIR/../daemon_pid.fl
fi

echo $BASHPID > $DIR/../daemon_pid.fl

while true
do
  sleep 10
  if [[ ! -f "$DIR/../pid.fl" ]]; then
    echo "start fl because of no pid.fl"
    start_fl
    continue
  fi
  pid=`cat $DIR/../pid.fl`
  kill -0 ${pid} 2> /dev/null 1>&2
  if [[ $? -ne 0 ]]; then
    if [[ -f "$DIR/../shutdown.fl" ]]; then
      echo "Gracefully shutdown."
      break
    fi
    echo "start fl because process of ${pid} does not exist"
    start_fl
    continue
  fi
  if [[ -f "$DIR/../shutdown.fl" ]]; then
    echo "About to shutdown."
    stop_fl
    break
  fi
  if [[ -f "$DIR/../restart.fl" ]]; then
    echo "About to restart."
    stop_fl
  fi
done

rm -f $DIR/../pid.fl $DIR/../shutdown.fl $DIR/../restart.fl $DIR/../daemon_pid.fl
