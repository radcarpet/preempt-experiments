DEVICE=0
NER_MODEL=/PATH/TO/universal-ner-pii
QA_MODEL=gpt-4
STS_MODEL=/PATH/TO/all-mpnet-base-v2

NER_TAG=uniner
QA_TAG=gpt

ENTITY=Name
SAMPLES=10
START=0
END=50
API_KEY=API_KEY

TAG=./results/${NER_TAG}_ner_extract_${QA_TAG}_qa_${ENTITY}_${START}_${END}_${SAMPLES}

CUDA_VISIBLE_DEVICES=${DEVICE} python3 long_context_task.py \
    --entity ${ENTITY} \
    --qa_path ${QA_MODEL} \
    --batch_size 1 \
    --ner_path ${NER_MODEL} \
    --sts_path ${STS_MODEL} \
    --tag ${TAG} \
    --samples ${SAMPLES} \
    --api_key ${API_KEY} \
    --start ${START} \
    --end ${END} \
    --seed 24 \
    --use_fpe true \
    --metrics_only true \
    # --parse_data true \


ENTITY=Name
SAMPLES=10
START=10
END=50

TAG=./results/${NER_TAG}_ner_extract_${QA_TAG}_qa_${ENTITY}_${START}_${END}_${SAMPLES}

CUDA_VISIBLE_DEVICES=${DEVICE} python3 long_context_task.py \
    --entity ${ENTITY} \
    --qa_path ${QA_MODEL} \
    --batch_size 1 \
    --ner_path ${NER_MODEL} \
    --sts_path ${STS_MODEL} \
    --tag ${TAG} \
    --samples ${SAMPLES} \
    --api_key ${API_KEY} \
    --start ${START} \
    --end ${END} \
    --seed 22 \
    --use_fpe true \
    --metrics_only true \
    # --parse_data true \


ENTITY=Name
SAMPLES=10
START=20
END=50

TAG=./results/${NER_TAG}_ner_extract_${QA_TAG}_qa_${ENTITY}_${START}_${END}_${SAMPLES}

CUDA_VISIBLE_DEVICES=${DEVICE} python3 long_context_task.py \
    --entity ${ENTITY} \
    --qa_path ${QA_MODEL} \
    --batch_size 1 \
    --ner_path ${NER_MODEL} \
    --sts_path ${STS_MODEL} \
    --tag ${TAG} \
    --samples ${SAMPLES} \
    --api_key ${API_KEY} \
    --start ${START} \
    --end ${END} \
    --seed 20 \
    --use_fpe true \
    --metrics_only true \
    # --parse_data true \


ENTITY=Name
SAMPLES=10
START=30
END=50

TAG=./results/${NER_TAG}_ner_extract_${QA_TAG}_qa_${ENTITY}_${START}_${END}_${SAMPLES}

CUDA_VISIBLE_DEVICES=${DEVICE} python3 long_context_task.py \
    --entity ${ENTITY} \
    --qa_path ${QA_MODEL} \
    --batch_size 1 \
    --ner_path ${NER_MODEL} \
    --sts_path ${STS_MODEL} \
    --tag ${TAG} \
    --samples ${SAMPLES} \
    --api_key ${API_KEY} \
    --start ${START} \
    --end ${END} \
    --seed 22 \
    --use_fpe true \
    --metrics_only true \
    # --parse_data true \


ENTITY=Name
SAMPLES=10
START=40
END=50

TAG=./results/${NER_TAG}_ner_extract_${QA_TAG}_qa_${ENTITY}_${START}_${END}_${SAMPLES}

CUDA_VISIBLE_DEVICES=${DEVICE} python3 long_context_task.py \
    --entity ${ENTITY} \
    --qa_path ${QA_MODEL} \
    --batch_size 1 \
    --ner_path ${NER_MODEL} \
    --sts_path ${STS_MODEL} \
    --tag ${TAG} \
    --samples ${SAMPLES} \
    --api_key ${API_KEY} \
    --start ${START} \
    --end ${END} \
    --seed 20 \
    --use_fpe true \
    --metrics_only true \
    # --parse_data true \