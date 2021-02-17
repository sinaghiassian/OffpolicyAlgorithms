from Registry.AlgRegistry import alg_dict

# noinspection SpellCheckingInspection
colors = ['black', "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
          "#17becf"]
color_dict = {alg_name: color for alg_name, color in zip(alg_dict.keys(), colors)}
# algs_groups = {'main_algs': ['TD', 'GTD', 'ETD'], 'gradients': ['GTD', 'GTD2', 'HTD', 'PGTD2', 'TDRC'],
#                'emphatics': ['ETD', 'ETDLB'], 'fast_algs': ['TD', 'TB', 'Vtrace', 'ABTD']}
algs_groups = {'gradients': ['GTD', 'GTD2', 'HTD', 'PGTD2', 'TDRC']}
exp_names = ['1HVFourRoom', 'FirstFourRoom', 'FirstChain']
