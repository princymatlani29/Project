from numpy.random import uniform, normal, randint, rand, choice
from numpy import pi, sin, cos, zeros, minimum, maximum, abs, where, sign, mean, stack, exp, any
from opfunu.cec_basic.cec2014_nobias import *
from numpy import min as np_min
from numpy import max as np_max
from copy import deepcopy
from root import Root
import numpy as np


class BaseMSA(Root):

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=False, epoch=100, pop_size=100,c1=1.2, c2=1.2, w_min=0.4, w_max=0.9, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.c1 = c1          
        self.c2 = c2
        self.w_min = w_min     
        self.w_max = w_max
    
    def get_fitness_position(self, position=None, minmax=0):
        return self.obj_func(position) if minmax == 0 else 1.0 / (self.obj_func(position) + self.EPSILON)

    def get_fitness_solution(self, solution=None, minmax=0):
        return self.get_fitness_position(solution[self.ID_POS], minmax)

    def get_global_best_solution(self, pop=None, id_fit=None, id_best=None):
        sorted_pop = sorted(pop, key=lambda temp: temp[id_fit])
        return deepcopy(sorted_pop[id_best])
    
    def create_solution(self,sel_data):
        position = sel_data
        fitness = self.get_fitness_position(position=position, minmax=1)
        return [position, fitness]

    def train(self,data):
        pop = [self.create_solution(data[_]) for _ in range(self.pop_size)]
        v_max = 0.5 * (self.ub - self.lb)
        v_min = zeros(self.problem_size)
        v_list = uniform(v_min, v_max, (self.pop_size, self.problem_size))
        pop_local = deepcopy(pop)
        g_best = self.get_global_best_solution(pop=pop, id_fit=self.ID_FIT, id_best=self.ID_MIN_PROB)
        gg = g_best[0]
        gg = np.sort(gg)

        for epoch in range(self.epoch):
            w = (self.epoch - epoch) / self.epoch * (self.w_max - self.w_min) + self.w_min
            for i in range(self.pop_size):
                
                best = gg[self.ID_POS]
                worst = gg[self.ID_FIT]
                m = (sum(pop[i][self.ID_POS])-worst)/(best-worst)
                Mt = 1-((epoch-1)/(self.epoch-1))
                r1=rand()
                
                Ud = r1*Mt*v_max[self.ID_POS]*sin(best-pop[i][self.ID_POS][self.ID_POS])
                Pd = Mt*Ud*epoch
                v_new = (2*Mt*Pd) / (m+Mt)
                r2 =rand()
                
                x_new = pop[i][self.ID_POS] + (r2*v_new)
                x_new = self.amend_position_random_faster(x_new)
                fit_new = self.get_fitness_position(x_new)
                pop[i] = [x_new, fit_new]

                if fit_new < pop_local[i][self.ID_FIT]:
                    pop_local[i] = [x_new, fit_new]

                ## batch size idea
                if self.batch_idea:
                    if (i + 1) % self.batch_size == 0:
                        g_best = self.update_global_best_solution(pop_local, self.ID_MIN_PROB, g_best)
                else:
                    if (i + 1) % self.pop_size:
                        g_best = self.update_global_best_solution(pop_local, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch+1, g_best[self.ID_FIT]))
        self.solution = g_best
        sg=self.solution[self.ID_POS]
        return  np.sort(sg)[self.ID_POS]

def optimize(data):
    obj_func = F5
    verbose = False
    epoch = 100;pop_size = 50
    data = rand(50, 10)
    problemSize = 10
    lb2 = -5;ub2 = 10
    md2 = BaseMSA(obj_func, lb2, ub2, verbose, epoch, pop_size, problem_size=problemSize)  
    best_p= md2.train(data)
    return format(best_p,".2f")

