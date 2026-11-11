import numpy as np, sys, json
from multiprocessing import Pool
sys.path.insert(0,'.')
from bifrontal_sim import *
NB=500
def ep(args):
    i,agent,kw=args
    cfg=Config(eps=0.1,**kw)
    b=make_board(np.random.default_rng(900000+i),8)   # tuning boards (disjoint from test seeds)
    return run_episode(b,cfg,agent,seed=950000+i)['outcome']
def util(outs):
    o=np.array(outs); return (o=='success').mean()-(o=='trap').mean(), (o=='success').mean(), (o=='trap').mean()
if __name__=='__main__':
    out={}
    with Pool(8) as p:
        for lam in [16,64,256,1024]:
            r=util(p.map(ep,[(i,'SP',dict(lam0=lam)) for i in range(NB)]))
            out[f'SP_lam{lam}']=r; print('SP lam0',lam,r,flush=True)
        best_lam=max([16,64,256,1024],key=lambda l: out[f'SP_lam{l}'][0])
        for tau in [0.05,0.1,0.2,0.3,0.4,0.5]:
            r=util(p.map(ep,[(i,'TW',dict(lam0=best_lam,tau0=tau)) for i in range(NB)]))
            out[f'TW_tau{tau}']=r; print('TW tau0',tau,r,flush=True)
    best_tau=max([0.05,0.1,0.2,0.3,0.4,0.5],key=lambda t: out[f'TW_tau{t}'][0])
    json.dump(dict(results=out,best_lam=best_lam,best_tau=best_tau),open('tuning.json','w'),indent=1)
    print('BEST',best_lam,best_tau)
