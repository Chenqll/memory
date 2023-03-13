import torch

def _get_gpu_mem():
    return torch.cuda.memory_allocated(),torch.cuda.memory_cached()


def _generate_mem_hook(idx,mem,hook_type):
    def hook(self,*args):
        mem_all,mem_cache=_get_gpu_mem()
        torch.cuda.synchronize()
        mem.append({
            'layer_idx':idx,
            'hook_type':hook_type,
            'mem_all':mem_all,
            'mem_cache':mem_cache
        })

    return hook

def _add_mem_hook(idx,mod,mem,hr):
    h=mod.register_forward_pre_hook(_generate_mem_hook(idx,mem,'pre'))
    hr.append(h)

    h=mod.register_forward_hook(_generate_mem_hook(idx,mem,'fwd'))
    hr.append(h)

    h=mod.register_backward_hook(_generate_mem_hook(idx,mem,'bwd'))
    hr.append(h)
    


def log_mem(model,inp,mem_log=[],exp=None):
    

    mem_log=[]

    exp=exp
    hr=[]

    for idx,module in enumerate(model.modules()):
        _add_mem_hook(idx,module,mem_log,hr)

    try:
        out=model(inp)
        loss=out.sum()
        loss.backward()
    finally:
        [h.remove for h in hr]
    return mem_log