DEVICE=0
NER_TAG=uniner
NER_MODEL=/PATH/TO/uniner-7b-pii

TASK=de-en
ENTITY=Money
TRANS_MODEL=opus-mt-de-en
# TRANS_MODEL=gpt-4
SAMPLES=50
API_KEY=SECRET_KEY

TAG=./results/${NER_TAG}_ner_extract_${TRANS_MODEL}_translation_${TASK}_${ENTITY}_FPE_UPDATED

CUDA_VISIBLE_DEVICES=${DEVICE} python3 translation_task.py \
    --task ${TASK} \
    --entity ${ENTITY} \
    --translation_model ${TRANS_MODEL} \
    --batch_size 1 \
    --ner_path ${NER_MODEL} \
    --tag ${TAG} \
    --samples ${SAMPLES} \
    --api_key ${API_KEY} \
    --use_fpe true \