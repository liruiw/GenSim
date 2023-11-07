DATA_DIR=$1
TASK=$2
DISP=False

echo "Generating dataset... Folder: $DATA_DIR"
trap "kill 0" SIGINT

LANG_TASKS=$2

for task in $LANG_TASKS
    do
        python cliport/demos.py n=200  task=$task mode=train data_dir=$DATA_DIR disp=$DISP record.save_video=False  +regenerate_data=True &
        python cliport/demos.py n=50   task=$task mode=val   data_dir=$DATA_DIR disp=$DISP record.save_video=False   +regenerate_data=True &
        python cliport/demos.py n=100  task=$task mode=test  data_dir=$DATA_DIR disp=$DISP record.save_video=False   +regenerate_data=True &
    done
wait

echo "Finished Language Tasks."