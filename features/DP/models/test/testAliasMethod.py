'''aliasMethod非常接近
'''
import numpy as np

import sys
sys.path.append('../')

from edgeDP import AliasMethod


def gen_prob_dist(N):
    p = np.random.randint(0,100,N)
    return p/np.sum(p)

def simulate(N=100,k=100000,):
    truth = gen_prob_dist(N)
    aliasMethod = AliasMethod(truth=truth)
    
    ans = np.zeros(N)
    for _ in range(k):
        i = aliasMethod.sample()
        ans[i] += 1
    return ans/np.sum(ans),truth

if __name__ == "__main__":
    alias_result, truth = simulate()

    for i in range(len(alias_result)):
        print(i, alias_result[i], truth[i])

