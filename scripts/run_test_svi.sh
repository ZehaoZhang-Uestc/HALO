python train_HALO_sample.py --val --pretrain saved_models/best.pth.tar  --data_name CVOGL_SVI --savename test_model_svi  --batch_size 1 --num_workers 16 --print_freq 50
python train_HALO_sample.py --test --pretrain saved_modelsbest.pth.tar  --data_name CVOGL_SVI --savename test_model_svi  --batch_size 1 --num_workers 16 --print_freq 50
