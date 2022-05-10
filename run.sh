# Train the model
python re_main.py --do_train --task_name semeval --train_data_path=./data/train.tsv --dev_data_path=./data/dev.tsv --use_bert --bert_model ./pre-training/models/demo/final --train_batch_size 16 --gradient_accumulation_steps=2 --eval_batch_size 32 --max_seq_length 500 --num_train_epochs 2 --learning_rate 1e-5 --warmup_proportion 0.2 --patient 100 --seed=42 --model_name demo

# Test the model
python re_main.py --do_test --task_name semeval --test_data_path=./data/test.tsv --eval_model=./models/demo/model --eval_batch_size 4

