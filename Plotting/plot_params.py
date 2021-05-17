from Plotting.plot_utils import FirstChainAttr, FirstFourRoomAttr, HVFirstFourRoomAttr
from Registry.AlgRegistry import alg_dict


PLOT_RERUN = True
PLOT_RERUN_AND_ORIG = False
if PLOT_RERUN and PLOT_RERUN_AND_ORIG:
    PLOT_RERUN_AND_ORIG = False
RERUN_POSTFIX = '_rerun'
DEBUG_MODE = True

# noinspection SpellCheckingInspection
COLORS = ['#000000', "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
          "#17becf"]
ALG_COLORS = {alg_name: color for alg_name, color in zip(alg_dict.keys(), COLORS)}
ALG_COLORS['LSTD'] = ALG_COLORS['TD']
ALG_COLORS['LSETD'] = ALG_COLORS['ETD']
ALG_GROUPS = {'main_algs': ['TD', 'GTD', 'ETD', 'LSTD', 'LSETD'],
              'gradients': ['GTD', 'GTD2', 'HTD', 'PGTD2', 'TDRC', 'LSTD'],
              'emphatics': ['ETD', 'ETDLB', 'LSETD'],
              'fast_algs': ['TD', 'TB', 'Vtrace', 'ABTD', 'LSTD']}
EXPS = ['1HVFourRoom', 'FirstFourRoom', 'FirstChain']
ALGS = [key for key in alg_dict.keys()]
ALGS.remove('LSTD')
ALGS.remove('LSETD')
# ALGS.remove('TDRC')
ALL_ALGS = ['TD', 'GTD', 'GTD2', 'PGTD2', 'HTD', 'TDRC', 'ETD', 'ETDLB', 'TB', 'Vtrace', 'ABTD', 'LSTD', 'LSETD']
# ALL_ALGS = ['TD', 'Vtrace', 'TB', 'ABTD']
LMBDA_AND_ZETA = [0.0, 0.9]
AUC_AND_FINAL = ['auc', 'final']
EXP_ATTRS = {'FirstChain': FirstChainAttr, 'FirstFourRoom': FirstFourRoomAttr, '1HVFourRoom': HVFirstFourRoomAttr}

if DEBUG_MODE:
    EXPS = ['FirstFourRoom', '1HVFourRoom']
    # ALGS = ['GTD']
    # ALL_ALGS.remove('ETDLB')
    # ALL_ALGS.remove('LSTD')
    # ALL_ALGS.remove('LSETD')
    # LMBDA_AND_ZETA = [0.9]
    AUC_AND_FINAL = ['final']
    # ALG_GROUPS = {'main_algs': ALL_ALGS}
