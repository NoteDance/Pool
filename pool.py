import numpy as np
import multiprocessing as mp
import math


class Pool:
    def __init__(self, env, processes, pool_size, window_size=None, clearing_freq=None, window_size_=None, random=True):
        self.env = env
        self.processes = processes
        self.pool_size = pool_size
        self.window_size = window_size
        self.clearing_freq = clearing_freq
        self.window_size_ = window_size_
        self.random = random
        manager=mp.Manager()
        self.state_pool_list=manager.list()
        self.action_pool_list=manager.list()
        self.next_state_pool_list=manager.list()
        self.reward_pool_list=manager.list()
        self.done_pool_list=manager.list()
        self.inverse_len=manager.list([0 for _ in range(processes)])
        self.lock_list=[mp.Lock() for _ in range(self.processes)]
        if self.clearing_freq!=None:
            self.store_counter=manager.list()
    
    def pool(self,s,a,next_s,r,done,index=None):
        if self.state_pool_list[index] is None:
            self.state_pool_list[index]=s
            self.action_pool_list[index]=np.expand_dims(a,axis=0)
            self.next_state_pool_list[index]=np.expand_dims(next_s,axis=0)
            self.reward_pool_list[index]=np.expand_dims(r,axis=0)
            self.done_pool_list[index]=np.expand_dims(done,axis=0)
        else:
            self.state_pool_list[index]=np.concatenate((self.state_pool_list[index],s),0)
            self.action_pool_list[index]=np.concatenate((self.action_pool_list[index],np.expand_dims(a,axis=0)),0)
            self.next_state_pool_list[index]=np.concatenate((self.next_state_pool_list[index],np.expand_dims(next_s,axis=0)),0)
            self.reward_pool_list[index]=np.concatenate((self.reward_pool_list[index],np.expand_dims(r,axis=0)),0)
            self.done_pool_list[index]=np.concatenate((self.done_pool[7],np.expand_dims(done,axis=0)),0)
        if self.clearing_freq!=None:
            self.store_counter[index]+=1
            if self.store_counter[index]%self.clearing_freq==0:
                self.state_pool_list[index]=self.state_pool_list[index][self.window_size_:]
                self.action_pool_list[index]=self.action_pool_list[index][self.window_size_:]
                self.next_state_pool_list[index]=self.next_state_pool_list[index][self.window_size_:]
                self.reward_pool_list[index]=self.reward_pool_list[index][self.window_size_:]
                self.done_pool_list[index]=self.done_pool_list[index][self.window_size_:]
        if len(self.state_pool_list[index])>math.ceil(self.pool_size/self.processes):
            if self.window_size!=None:
                self.state_pool_list[index]=self.state_pool_list[index][self.window_size:]
                self.action_pool_list[index]=self.action_pool_list[index][self.window_size:]
                self.next_state_pool_list[index]=self.next_state_pool_list[index][self.window_size:]
                self.reward_pool_list[index]=self.reward_pool_list[index][self.window_size:]
                self.done_pool_list[index]=self.done_pool_list[index][self.window_size:]
            else:
                self.state_pool_list[index]=self.state_pool_list[index][1:]
                self.action_pool_list[index]=self.action_pool_list[index][1:]
                self.next_state_pool_list[index]=self.next_state_pool_list[index][1:]
                self.reward_pool_list[index]=self.reward_pool_list[index][1:]
                self.done_pool_list[index]=self.done_pool_list[index][1:]
    
    def store_in_parallel(self,p,lock_list):
        s,a=self.env[p].reset()
        s=np.array(s)
        while True:
            if self.random==True:
                if self.state_pool_list[p] is None:
                    index=p
                    self.inverse_len[index]=1
                else:
                    total_inverse=np.sum(self.inverse_len)
                    prob=self.inverse_len/total_inverse
                    index=np.random.choice(self.processes,p=prob.numpy())
                    self.inverse_len[index]=1/(len(self.state_pool_list[index])+1)
            else:
                index=p
            s=np.expand_dims(s,axis=0)
            a=self.select_action(s)
            a,next_s,r,done=self.env[p].step(a)
            next_s=np.array(next_s)
            r=np.array(r)
            done=np.array(done)
            lock_list[index].acquire()
            self.pool(s,a,next_s,r,done,index)
            lock_list[index].release()
            if done:
                return
            s=next_s
    
    def store(self):
        process_list=[]
        for p in range(self.processes):
            process=mp.Process(target=self.store_in_parallel,args=(p,self.lock_list))
            process.start()
            process_list.append(process)
        for process in process_list:
            process.join()
    
    def get_pool(self):
        state_pool=np.concatenate(self.state_pool_list)
        action_pool=np.concatenate(self.action_pool_list)
        next_state_pool=np.concatenate(self.next_state_pool_list)
        reward_pool=np.concatenate(self.reward_pool_list)
        done_pool=np.concatenate(self.done_pool_list)
        return state_pool, action_pool, next_state_pool, reward_pool, done_pool
