import seaborn as sns; sns.set()
import pandas as pd
import matplotlib.pyplot as plt
import json
import sys, os, time
import os.path as osp
import numpy as np
import yaml
from pathlib import Path
pd.set_option("display.max_rows",200)
# 设置
tot_step = 10**7
dur_step = 80000
num_step = int(tot_step/dur_step)
pd.options.display.notebook_repr_html=False  # 表格显示
plt.rcParams['figure.dpi'] = 100  # 图形分辨率

def data_statistic(map_address,map_name):
    data = []
    for i in range(1,2):
        address = osp.join(map_address,map_name)
        data.append(Path(address))
    return data 
  
def smooth(data, sm=1):
    if sm > 1:
        smooth_data = []
        for d in data:
            y = np.ones(sm)*1.0/sm
            d = np.convolve(y, d, "same")
 
            smooth_data.append(d)
 
    return smooth_data

def extrwithdata(json_paths, location):
    step = 1
    for path in json_paths:
        with open(path, 'r') as f:
            json_data = json.load(f)
            # print("json_data",json_data)
            data = np.array(json_data[location])
        if(step == 1):
            save_data = data;
        else:
            save_data = np.vstack((save_data,data))
        step += 1
    #print(" save_data", save_data)
    #print(" save_data", len(save_data))
    return save_data
    
def dealwithdata(save_data, num_result, Timesteps, value):
    for i in range(1,num_result+1):
        if i==1:
            data_i= save_data[0][:,2]
        else:
            data_i = data_i + save_data[i-1][:,2]
    data_mean = (data_i)/num_result
    for i in range(1,num_result+1):
        if i==1:
            data_i= (save_data[0][:,2]-data_mean)**2
        else:
            data_i = data_i + (save_data[i-1][:,2]-data_mean)**2
    data_var = np.sqrt((data_i)/num_result)
    data = [data_mean, data_mean+data_var, data_mean-data_var]
    data = pd.DataFrame(smooth(data,sm=2)).melt(var_name=Timesteps,value_name= value)
    data[Timesteps] = data[Timesteps] * dur_step 
    data[value] =  data[value] 
    data.to_csv("data.csv")
    return data


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', nargs='*')
    parser.add_argument('--xaxis', default='Epoch')
    parser.add_argument('--x', default='TotalEnvInteracts')
    parser.add_argument('--y', default='AverageEpRet')
    parser.add_argument('--condition', '-c', default='Method')
    parser.add_argument('--cut', type=int, default=0)
    parser.add_argument('--hline', type=float, nargs='*')
    parser.add_argument('--linename', nargs='*')
    args = parser.parse_args()
    

    num_result = 5
    dir_mujoco_results = "/home/dell/zlj/code/AI2023/MACPO/macpo/scripts/results/mujoco"
    rewards = "Average Episode Reward"
    costs = "Average Episode Cost"
    Timesteps = "Environment steps"
    
##  获取并处理agent_conf_6_1保存在的summary.json中的rewards与costs数据  
    Rewards_agent_conf_6_1 =[[0 for col in range(num_step)] for row in range(num_result)] 
    Costs_agent_conf_6_1 =[[0 for col in range(num_step)] for row in range(num_result)] 
    for i in range(1,num_result+1):
        j = i 
        dir_agent_conf_6_1 = dir_mujoco_results + "/manyagent_ant/macpo/agent_conf-6-1/rnn/" 
        data_agent_conf_6_1 = "run"+ str(j) + "/logs/summary.json"
        rewards_loc_agent_conf_6_1 = dir_agent_conf_6_1 + "run"+ str(j) + "/logs/train_episode_rewards/aver_rewards"
        costs_loc_agent_conf_6_1 = dir_agent_conf_6_1 + "run"+ str(j) + "/logs/train_episode_costs/aver_costs"
        results_agent_conf_6_1 = data_statistic(dir_agent_conf_6_1, data_agent_conf_6_1)
        
        ##  提取rewards与costs数据
        rewards_agent_conf_6_1 = extrwithdata(results_agent_conf_6_1,rewards_loc_agent_conf_6_1)
        costs_agent_conf_6_1 = extrwithdata(results_agent_conf_6_1, costs_loc_agent_conf_6_1)
        Rewards_agent_conf_6_1[i-1] = rewards_agent_conf_6_1
        Costs_agent_conf_6_1[i-1] = costs_agent_conf_6_1
    Rewards_agent_conf_6_1 = np.array(Rewards_agent_conf_6_1)
    Costs_agent_conf_6_1 = np.array(Costs_agent_conf_6_1) 
    np.save(file = dir_agent_conf_6_1 + "rewards_data.npy", arr = Rewards_agent_conf_6_1)
    np.save(file = dir_agent_conf_6_1 + "costs_data.npy", arr = Costs_agent_conf_6_1)
    
    #处理rewards与costs数据
    Rewards_agent_conf_6_1 = dealwithdata(Rewards_agent_conf_6_1, num_result, Timesteps, rewards)
    Costs_agent_conf_6_1 = dealwithdata(Costs_agent_conf_6_1, num_result, Timesteps, costs)


    
##  获取并处理agent_conf_3_2保存在的summary.json中的rewards与costs数据
    Rewards_agent_conf_3_2 =[[0 for col in range(num_step)] for row in range(num_result)] 
    Costs_agent_conf_3_2 =[[0 for col in range(num_step)] for row in range(num_result)] 
    for i in range(1,num_result+1):
        j = i 
        dir_agent_conf_3_2 = dir_mujoco_results + "/manyagent_ant/macpo/agent_conf-3-2/rnn/" 
        data_agent_conf_3_2 = "run"+ str(j) + "/logs/summary.json"
        rewards_loc_agent_conf_3_2 = dir_agent_conf_3_2 + "run"+ str(j) + "/logs/train_episode_rewards/aver_rewards"
        costs_loc_agent_conf_3_2 = dir_agent_conf_3_2 + "run"+ str(j) + "/logs/train_episode_costs/aver_costs"
        results_agent_conf_3_2 = data_statistic(dir_agent_conf_3_2, data_agent_conf_3_2)
        
        ##  提取rewards与costs数据
        rewards_agent_conf_3_2 = extrwithdata(results_agent_conf_3_2,rewards_loc_agent_conf_3_2)
        costs_agent_conf_3_2 = extrwithdata(results_agent_conf_3_2, costs_loc_agent_conf_3_2)
        Rewards_agent_conf_3_2[i-1] = rewards_agent_conf_3_2
        Costs_agent_conf_3_2[i-1] = costs_agent_conf_3_2
    Rewards_agent_conf_3_2 = np.array(Rewards_agent_conf_3_2)
    Costs_agent_conf_3_2 = np.array(Costs_agent_conf_3_2) 
    np.save(file = dir_agent_conf_3_2 + "rewards_data.npy", arr = Rewards_agent_conf_3_2)
    np.save(file = dir_agent_conf_3_2 + "costs_data.npy", arr = Costs_agent_conf_3_2)  
    
    #处理rewards与costs数据
    Rewards_agent_conf_3_2 = dealwithdata(Rewards_agent_conf_3_2, num_result, Timesteps, rewards)
    Costs_agent_conf_3_2 = dealwithdata(Costs_agent_conf_3_2, num_result, Timesteps, costs)

    
      
##  获取并处理agent_conf_2_3保存在的summary.json中的rewards与costs数据
    num_result = 5
    
    Rewards_agent_conf_2_3 =[[0 for col in range(num_step)] for row in range(num_result)] 
    Costs_agent_conf_2_3 =[[0 for col in range(num_step)] for row in range(num_result)] 
    for i in range(1,num_result+1):
        j = i 
        dir_agent_conf_2_3 = dir_mujoco_results + "/manyagent_ant/macpo/agent_conf-6-1/rnn/" 
        data_agent_conf_2_3 = "run"+ str(j) + "/logs/summary.json"
        rewards_loc_agent_conf_2_3 = dir_agent_conf_2_3 + "run"+ str(j) + "/logs/train_episode_rewards/aver_rewards"
        costs_loc_agent_conf_2_3 = dir_agent_conf_2_3 + "run"+ str(j) + "/logs/train_episode_costs/aver_costs"
        results_agent_conf_2_3 = data_statistic(dir_agent_conf_2_3, data_agent_conf_2_3)
        
        ##  提取rewards与costs数据
        rewards_agent_conf_2_3 = extrwithdata(results_agent_conf_2_3, rewards_loc_agent_conf_2_3)
        costs_agent_conf_2_3 = extrwithdata(results_agent_conf_2_3, costs_loc_agent_conf_2_3)
        Rewards_agent_conf_2_3[i-1] = rewards_agent_conf_2_3
        Costs_agent_conf_2_3[i-1] = costs_agent_conf_2_3
    Rewards_agent_conf_2_3 = np.array(Rewards_agent_conf_2_3)
    Costs_agent_conf_2_3 = np.array(Costs_agent_conf_2_3) 
    np.save(file = dir_agent_conf_2_3 + "rewards_data.npy", arr = Rewards_agent_conf_2_3)
    np.save(file = dir_agent_conf_2_3 + "costs_data.npy", arr = Costs_agent_conf_2_3)     
    
    #处理rewards与costs数据
    Rewards_agent_conf_2_3 = dealwithdata(Rewards_agent_conf_2_3, num_result, Timesteps, rewards)
    Costs_agent_conf_2_3 = dealwithdata(Costs_agent_conf_2_3, num_result, Timesteps, costs)

    
    # Figure
    fig,ax = plt.subplots(2,3,sharey=True)
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.2, hspace=0.2)
    sns.set_style('whitegrid')
    plot = plt.title("ManyAgent Ant")
    plt.subplot(231)
    sns.lineplot(data=Costs_agent_conf_2_3, x=Timesteps, y=costs)
    plt.ylabel(costs)
    plt.subplot(232)
    sns.lineplot(data=Costs_agent_conf_3_2, x=Timesteps, y=costs)
    plt.ylabel("")
    plt.subplot(233)
    sns.lineplot(data=Costs_agent_conf_6_1, x=Timesteps, y=costs,label="MAPPO-L")
    plt.ylabel("")
    plt.subplot(234)
    sns.lineplot(data=Rewards_agent_conf_2_3, x=Timesteps, y=rewards)
    plt.ylabel(rewards)
    plt.subplot(235)
    sns.lineplot(data=Rewards_agent_conf_3_2, x=Timesteps, y=rewards)
    plt.ylabel("")
    plt.subplot(236)
    sns.lineplot(data=Rewards_agent_conf_6_1, x=Timesteps, y=rewards, label="MAPPO-L")
    plt.ylabel("")
    
    plt.show()
    fig = plot.get_figure()
    
    # 这一步保存图片到一个绝对路径需要的话可以改，导出的图片可以增强图片分辨率
    fig.savefig(dir_mujoco_results +"result.jpg", dpi=1080) 
    fig.savefig(dir_mujoco_results +"result.png", dpi=1080) 
    fig.savefig(dir_mujoco_results +"result.pdf", dpi=1080)
    fig.savefig(dir_mujoco_results +"result.eps", dpi=1080)
    
    fig.savefig(dir_mujoco_results +"result-tight.jpg", dpi=1080, bbox_inches="tight") 
    fig.savefig(dir_mujoco_results +"result-tight.png", dpi=1080, bbox_inches="tight") 
    fig.savefig(dir_mujoco_results +"result-tight.pdf", dpi=1080, bbox_inches="tight")
    fig.savefig(dir_mujoco_results +"result-tight.eps", dpi=1080, bbox_inches="tight")
 
if __name__ == "__main__":
    main()
