import os, yaml

cfg_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)

SEED        = cfg['seed']
TEST_SIZE   = cfg['test_size']
GRID_K_FOLD = cfg['grid_k_fold']
LR_MAXITER  = cfg['lr_max_iter']
RF_N_EST    = cfg['rf_n_estimators']
RF_GRID     = cfg['rf_grid']
XGB_EVAL_MT = cfg['xgb_eval_metric']
XGB_GRID    = cfg['xgb_grid']
BIN_COLS    = cfg['bin_cols'] 

SELECTED_FEATURES = cfg['selected_features']
DEFAULT_TARGET    = cfg['default_target']