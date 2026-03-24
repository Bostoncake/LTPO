export HUGGING_FACE_TOKEN=<YOUR_HF_TOKEN>
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

export CUDA_VISIBLE_DEVICES=0

# dataset="mmvp"
# model=/WillDevExt/xiongyizhe/models/Qwen2.5-VL-7B-Instruct
# image_root=.
# max_new_tokens=2048
# max_num_steps=10
# num_thought_tokens=8
# sigma=4
# sigma_decay=0.9
# lr=0.04
# topk=10
# verbose=1

# python main_vl.py \
#     --dataset $dataset \
#     --data_root mllm_data \
#     --image_root $image_root \
#     --model_name_or_path $model \
#     --output_dir ./output \
#     --device cuda \
#     --max_new_tokens $max_new_tokens \
#     --max_num_steps $max_num_steps \
#     --num_thought_tokens $num_thought_tokens \
#     --sigma $sigma \
#     --sigma_decay $sigma_decay \
#     --lr $lr \
#     --top_k $topk \
#     --eval_baseline \
#     --verbose $verbose


# for dataset in "mmstar" "mm_math" "math_vista" "math_vision" "hallusion"; do
#     model=/WillDevExt/xiongyizhe/models/Qwen2.5-VL-7B-Instruct
#     image_root=.
#     max_new_tokens=2048
#     max_num_steps=10
#     num_thought_tokens=8
#     sigma=4
#     sigma_decay=0.9
#     lr=0.04
#     topk=10
#     verbose=1

#     python main_vl.py \
#         --dataset $dataset \
#         --data_root mllm_data \
#         --image_root $image_root \
#         --model_name_or_path $model \
#         --output_dir ./output \
#         --device cuda \
#         --max_new_tokens $max_new_tokens \
#         --max_num_steps $max_num_steps \
#         --num_thought_tokens $num_thought_tokens \
#         --sigma $sigma \
#         --sigma_decay $sigma_decay \
#         --lr $lr \
#         --top_k $topk \
#         --eval_baseline \
#         --verbose $verbose
# done

for dataset in "scienceqa"; do
    model=/WillDevExt/xiongyizhe/models/Qwen2.5-VL-7B-Instruct
    image_root=.
    max_new_tokens=2048
    max_num_steps=10
    num_thought_tokens=8
    sigma=4
    sigma_decay=0.9
    lr=0.04
    topk=10
    verbose=1

    python main_vl.py \
        --dataset $dataset \
        --data_root mllm_data \
        --image_root $image_root \
        --model_name_or_path $model \
        --output_dir ./output \
        --device cuda \
        --max_new_tokens $max_new_tokens \
        --max_num_steps $max_num_steps \
        --num_thought_tokens $num_thought_tokens \
        --sigma $sigma \
        --sigma_decay $sigma_decay \
        --lr $lr \
        --top_k $topk \
        --eval_baseline \
        --verbose $verbose
done