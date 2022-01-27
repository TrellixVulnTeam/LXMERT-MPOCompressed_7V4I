bash run/vqa_test.bash 0 vqa_lxr955_results --test test --mpo_layer="" --load snap/vqa/vqa_lxr955/BEST
bash run/vqa_test.bash 0 vqa_lxr955_results --test minival --mpo_layer="" --load snap/vqa/vqa_lxr955/BEST
bash run/vqa_test.bash 0 vqa_lxr955_l_results --test test --mpo_layer="l_layer" --load snap/vqa/vqa_lxr955_l/BEST
bash run/vqa_test.bash 0 vqa_lxr955_l_r_results --test test --mpo_layer="l_layer,r_layer" --load snap/vqa/vqa_lxr955_l_r/BEST
bash run/vqa_test.bash 0 vqa_lxr955_l_r_x_results --test test --mpo_layer="l_layer,r_layer,x_layer" --load snap/vqa/vqa_lxr955_l_r_x/BEST
bash run/vqa_test.bash 0 vqa_lxr955_l_r_x_emb_results --test test --mpo_layer="l_layer,r_layer,x_layer,embedding" --load snap/vqa/vqa_lxr955_l_r_x_emb/BEST
vqa.model.lxrt_encoder.model.bert.encoder.layer[0]

nohup bash run/vqa_finetune.bash 0 l_400 --mpo_layer="l_layer" --l_trunc_num=400 >> l_400.log 2>&1 &
nohup bash run/vqa_finetune.bash 1 baseline --mpo_layer="" >> baseline.log 2>&1 &
nohup bash run/vqa_finetune.bash 0 l_10 --mpo_layer="l_layer" --l_trunc_num=10 >> l_10.log 2>&1 &
nohup bash run/vqa_finetune.bash 0 r_400 --mpo_layer="r_layer" --r_trunc_num=400 >> r_400.log 2>&1 &

nohup bash run/vqa_finetune.bash 5 vqa_lxr955_freeze_l_r --mpo_layer="" >> vqa_f_l_r.log 2>&1 &