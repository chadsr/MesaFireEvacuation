from mesa.time import BaseScheduler
import pathos.pools as pp
import os
import random


class MultithreadedRandomActivation(BaseScheduler):
    def agent_step(self, agent):
        agent.step()

    def step(self):
        random.shuffle(self.agents)
        pool = pp.ProcessPool(os.cpu_count())
        pool.map(self.agent_step, self.agents)
        pool.close()

        self.steps += 1
        self.time += 1
