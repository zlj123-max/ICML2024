#!/bin/sh
env="mujoco"
scenario="manyagent_ant"
agent_conf="3x2"
agent_obsk=1
algo="macpo"
exp="rnn"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_mujoco.py  --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario ${scenario} --agent_conf ${agent_conf} --agent_obsk ${agent_obsk} --lr 9e-5 --critic_lr 5e-3 --std_x_coef 1 --std_y_coef 5e-1 --seed 50 --n_training_threads 4 --n_rollout_threads 16 --num_mini_batch 40 --episode_length 1000 --num_env_steps 10000000 --ppo_epoch 1 --use_value_active_masks  --add_center_xy --use_state_agent --kl_threshold 0.0065 --safety_bound 0.2 --safety_gamma 0.09 --line_search_fraction 0.5 --fraction_coef 0.27
done
