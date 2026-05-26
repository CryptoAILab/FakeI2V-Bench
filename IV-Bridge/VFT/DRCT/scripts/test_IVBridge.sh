# default
MODEL_NAME=convnext_base_in22k
MODEL_PATH=/data/lp/IVBridge-version-usenix/IVBridge-main/DRCT/output/weights/10_acc0.9831.pth
DEVICE_ID=0
EMBEDDING_SIZE=1024
MODEL_NAME=${1:-$MODEL_NAME}
MODEL_PATH=${2:-$MODEL_PATH}
DEVICE_ID=${3:-$DEVICE_ID}
EMBEDDING_SIZE=${4:-$EMBEDDING_SIZE}
ROOT_PATH=/data2/lp/IVBridge_datasets/crop/test/
FAKE_ROOT_PATH=""
DATASET_NAME=IVBridge
SAVE_TXT=../output/2convnext_results/GenVidBench.txt
INPUT_SIZE=224
BATCH_SIZE=24
FAKE_INDEXES=(1)
for FAKE_INDEX in ${FAKE_INDEXES[@]}
do
  echo FAKE_INDEX:${FAKE_INDEX},MODEL_NAME:${MODEL_NAME},MODEL_PATH:${MODEL_PATH}
  python train.py --root_path ${ROOT_PATH} --fake_root_path '' --model_name ${MODEL_NAME} \
                  --input_size ${INPUT_SIZE} --batch_size ${BATCH_SIZE} --device_id ${DEVICE_ID} --is_test \
                  --model_path ${MODEL_PATH} --is_crop --fake_indexes ${FAKE_INDEX} \
                  --dataset_name ${DATASET_NAME} --save_txt ${SAVE_TXT} --embedding_size ${EMBEDDING_SIZE}
done