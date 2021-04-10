
import matplotlib.pyplot as plt 
import numpy as np
import os
from opts import graph_save_dir
class graph(): #todo
    def __init__(self):
        self.training_loss=[]
        self.test_loss=[]
        self.save_name=os.path.join(graph_save_dir,"model_graph.png")
        self.first_time=False
    def update_graph(self,training_loss,test_loss):
        self.training_loss.append(training_loss)
        self.test_loss.append(test_loss)

    def draw_and_save(self):
        plt.plot( np.arange(0,len(self.training_loss)), self.training_loss,color='lightblue',linewidth=3,label='training_loss',marker='o', linestyle='dashed')
        plt.plot( np.arange(0,len(self.test_loss)), self.test_loss,color='darkgreen',linewidth=3,label='test_loss',marker='o', linestyle='dashed')
        if self.first_time==False:
            plt.xlabel("Epoch #")
            plt.ylabel("Total_loss")
            plt.legend()
            self.first_time = True
       # plt.show()
            plt.ylim(0,350)
            plt.xlim(0,100)
        plt.savefig(self.save_name)

def draw_graph():
    path = os.path.join(graph_save_dir,'graph_data.npy')
    graph_data = np.load(path)
    #np.append( graph_data,np.array([[tr_loss],[ts_loss]]),axis= 1)
    #grp =  graph()
   # print(graph_data.shape)
    save_name=os.path.join(graph_save_dir,"model_graph.png")
    
    plt.plot( np.arange(0,graph_data.shape[1]), graph_data[0,:],color='lightblue',linewidth=3,label='training_loss',marker='o', linestyle='dashed')
    plt.plot( np.arange(0,graph_data.shape[1]), graph_data[1,:],color='darkgreen',linewidth=3,label='test_loss',marker='o', linestyle='dashed')
    if graph_data.shape[1]==1:
        plt.xlabel("Epoch #")
        plt.ylabel("Total_loss")
        plt.legend()
        plt.ylim(0,15)
        plt.xlim(0,100)
    plt.savefig(save_name)
    #for i in range(graph_data.shape[1]):
#   grp.update_graph(graph_data[0][i],graph_data[1][i])
  #  grp.draw_and_save()


if __name__=='__main__':
    test_graph = graph()
    test_graph.update_graph(9,100)
    test_graph.update_graph(93,101)
    test_graph.update_graph(942,102)
    test_graph.update_graph(93,103)
    test_graph.update_graph(9,104)
    test_graph.draw_and_save()