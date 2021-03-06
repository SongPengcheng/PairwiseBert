python fine_tune.py \
  --model_name_or_path "bert-base-chinese" \
  --task_name "sim" \
  --data_dir "data/" \
  --do_train True \
  --do_eval False \
  --do_predict False \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --eval_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir "output/" \
  --save_total_limit 5 \
  --save_steps 1000 \
  --num_train_epochs 3