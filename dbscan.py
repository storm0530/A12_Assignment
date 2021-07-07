import math
import numpy as np
import matplotlib.pyplot as plt
import sys


DATA = [
    [0,0],
    [2,0],
    [2,1],
    [3,0],
    [4,1],
    [5,0],
    [6,0],
    [8,0]
    ]

EPS = 1 # eps parameter 
MINPTS = 2 # min points parameter
INIT_POINTS = [[2,10],[5,8],[1,2]]
K_NUM = 3
COLOR_CLUSTER = ["yellow","olive"]
COLOR_LIST = ["red","blue","black"]



# Function to calculate Euclidean distance between two points
def Euclidean_distance(x1,y1,x2,y2):
    d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return d

# Function to calculate Manhattan distance between two points
def Manhattan_distance(x1,y1,x2,y2):
    d = abs((x2 - x1)) + abs(y2 - y1)
    return d


def scan(data_p,input_list):
    '''
    data_p: 注目座標 例: [x,y]
    list: 注目座標が属するクラスタのリスト 例: [] or [ [x1,y1],[x2,y2],... ]
    返り値については距離計算とminptsの条件からlistに新たにデータを追加した
    input_listと，各ステップでの近傍座標リストnew_listを返す 例: [ [2,10],[5,8],[1,2] ]
    '''

    new_list = []

    for data_q in DATA:
        if Euclidean_distance(data_p[0],data_p[1],data_q[0],data_q[1]) <= EPS:
            new_list.append(data_q)
    
    if len(new_list) >= MINPTS:
        
        input_list.extend(new_list)
        numpy_input_list = np.array(input_list) # numpyに変換
        numpy_input_list = np.unique(numpy_input_list, axis=0) # 重複を削除
        input_list = numpy_input_list.tolist() # listに戻す
            
    return input_list, new_list


def cluster_search(cluster,data_p):
    '''
    暫定クラスタリストの中に，注目座標が含まれているかサーチする関数
    含まれていれば，暫定クラスタとそのindexを取り出す
    clusterの形は以下のようになっている
    [ 
        [ [2,3],[1,6] ],
        [ [1,-3],[0,4] ]
    ]
    '''
    for i in range(len(cluster)):
        if data_p in cluster[i]:
            return cluster[i], i
    
    return [], -1 # 暫定クラスタに含まれていなければindexは-1


# クラスタリングの可視化関数
def visualize(step,cluster,cluster_neighbor):
    '''
    step: DB Scanの何ステップ目か
    clusterの形は以下のようになっている
    [ 
        [ [2,3],[1,6] ],
        [ [1,-3],[0,4] ]
    ]
    cluster_neighbor: 各ステップでの近傍座標リスト 例:[[2,10],[5,8],[1,2]]
    '''

    fig = plt.figure()
    plt.title(f'Step{step+1}', fontsize=24)

    x = [DATA[i][0] for i in range(len(DATA))]
    y = [DATA[i][1] for i in range(len(DATA))]
    plt.scatter(x,y,s=50,c="black")

    cluster_neighbor_x = [cluster_neighbor[step][i][0] for i in range(len(cluster_neighbor[step]))]
    cluster_neighbor_y = [cluster_neighbor[step][i][1] for i in range(len(cluster_neighbor[step]))]
    plt.scatter(cluster_neighbor_x,cluster_neighbor_y,
    s=70,c="white",linewidths=2,edgecolors="black")
    
    for n in range(len(cluster)):
        if cluster[n] != []:
            cluster_x = [cluster[n][i][0] for i in range(len(cluster[n]))]
            cluster_y = [cluster[n][i][1] for i in range(len(cluster[n]))]
            plt.scatter(cluster_x, cluster_y, s=50, c=COLOR_CLUSTER[n])
    fig.savefig(f"./dbscan_visualization/result_step{step+1}.png")



cluster = []
for i in range(len(DATA)):
    cluster.append([])

cluster_neighbor = [] # 各ステップで近傍を出す際のもの
for i in range(len(DATA)):
    cluster_neighbor.append([])

step = 0
for data_p in DATA:
    print(f"DB Scanによるクラスタリング Step{step}")
    list, index = cluster_search(cluster, data_p)

    input_list = list
    output_list, output_neighbor = scan(data_p, input_list)
    cluster_neighbor[step] = output_neighbor
    if index != -1:
        cluster[index] = output_list
    else:
        for i in range(len(cluster)):
            if cluster[i] == []:
                cluster[i] = output_list
                break

    visualize(step,cluster,cluster_neighbor)
    step += 1


fig = plt.figure()
plt.title(f'Step Final', fontsize=24)

x = [DATA[i][0] for i in range(len(DATA))]
y = [DATA[i][1] for i in range(len(DATA))]
plt.scatter(x,y,s=50,c="black")

for n in range(len(cluster)):
    if cluster[n] != []:
        cluster_x = [cluster[n][i][0] for i in range(len(cluster[n]))]
        cluster_y = [cluster[n][i][1] for i in range(len(cluster[n]))]
        plt.scatter(cluster_x, cluster_y, s=50, c=COLOR_CLUSTER[n])

# 手動で点の色を変更
plt.scatter(DATA[0][0],DATA[0][1],s=50,c=COLOR_LIST[0])
plt.scatter(DATA[4][0],DATA[4][1],s=50,c=COLOR_LIST[1])
plt.scatter(DATA[7][0],DATA[7][1],s=50,c=COLOR_LIST[2])

fig.savefig(f"./dbscan_visualization/result_step_final.png")

# 手動で外れ値の点の色を変更
plt.scatter(DATA[0][0],DATA[0][1],
    s=70,c="white",linewidths=2,edgecolors="black")
plt.scatter(DATA[7][0],DATA[7][1],
    s=70,c="white",linewidths=2,edgecolors="red")

fig.savefig(f"./dbscan_visualization/result_step_final_outlier.png")