from graphviz import Digraph 
import torch

def plot_graph(codepath, filename='graph', directory='./',
               code2in = [0,1,0,1,2,1,2,3,2,3], code2out = [0,0,1,1,1,2,2,2,3,3]):
    ''' Plot the final searched model
    Args:
        codepath: path to the saved .pth file, generated from the searching script.
        arch_code_a: architecture code (decoded using model.decode).
        arch_code_c: cell operation code (decoded using model.decode).
        filename: filename to save graph.
        directory: directory to save graph.
        code2in, code2out: see definition in monai.networks.nets.dints.py.
    Return:
        graphviz graph.
    '''
    code = torch.load(codepath)
    arch_code_a = code['arch_code_a']
    arch_code_c = code['arch_code_c']
    ga = Digraph('G', filename=filename, engine='neato')
    depth = (len(code2in) + 2)//3
    # build a initial block
    inputs = []
    for _ in range(depth):
        inputs.append('(in,' + str(_) + ')')
    with ga.subgraph(name='cluster_all') as g:
        with g.subgraph(name='cluster_init') as c:
            for idx, _ in enumerate(inputs):
                c.node(_,pos='0,'+str(depth-idx)+'!')
        for blk_idx in range(arch_code_a.shape[0]):
            with g.subgraph(name='cluster'+str(blk_idx)) as c:
                outputs = [str((blk_idx,_)) for _ in range(depth)]
                for idx, _ in enumerate(outputs):
                    c.node(_,pos=str(2+2*blk_idx)+','+str(depth-idx)+'!')           
                for res_idx, activation in enumerate(arch_code_a[blk_idx]):
                    if activation:
                        c.edge(inputs[code2in[res_idx]], outputs[code2out[res_idx]], \
                            label=str(arch_code_c[blk_idx][res_idx]))          
                inputs = outputs
    ga.render(filename=filename, directory=directory, cleanup=True, format='png')
    return ga