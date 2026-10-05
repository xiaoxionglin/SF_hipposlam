import sys
import torch
import torch.nn.functional as F
import itertools

from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.utils.utils import log

from sf_working_directories.zeynep.dmlab.train_hipposlam import parse_dmlab_args, register_dmlab_components

def check_weight_collapse():
    # 1. EXACT ARGUMENTS FROM YOUR RUN
    mapname = "ymaze_instr_hl"
    #expname = "03_Q_grid_HighLevelRNN_see_1111_D.c.mod_FiLM_c.i.coe_200"
    expname = "07_Q_grid_HighLevelRNN_see_2222_D.c.mod_FiLM_c.i.coe_200"
    #expname = '13_Q_grid_HighLevelRNN_see_4444_D.c.mod_additive_c.i.coe_200'
    traindir = "/work/classic/fr_ze12-data/ymaze_newRNN/HighLevelChoice/stage2/Q_run6_grid_fixedrew_decdiff/train_dir/Q_grid_HighLevelRNN/Q_grid_HighLevelRNN_"
    
    cli = [
        "--algo", "APPO",
        "--env", mapname,
        "--experiment", expname,
        "--train_dir", traindir,
        "--core_name", "BypassSS_HighLevelRNN",
        "--Decoder_context_mod", "FiLM",
        "--context_injection_coef", "200",
        "--hl_K", "4",
        "--hl_d_H", "16",
        "--rnn_size", "1166",
        "--no_render",
        "--load_checkpoint_kind", "latest"
    ]

    cli_dict = {
        'algo': 'APPO',
        'env': mapname,
        'experiment': expname,
        'train_dir': traindir,
        'core_name': 'BypassSS_HighLevelRNN',
        'Decoder_context_mod': 'FiLM',
        "context_injection_coef": 200,
        'hl_K': 4,
        'hl_d_H': 16,
        'rnn_size': 1166,
        'no_render': True,
        'load_checkpoint_kind': 'latest'
    }

    # 2. LOAD CONFIGURATION
    try:
        register_dmlab_components()
    except Exception:
        pass # Handle safely if it expects cfg

    cfg = parse_dmlab_args(evaluation=True, argv=cli)
    cfg.cli_args = cli_dict
    
    # If the first register failed, do it safely here with the cfg
    try:
        register_dmlab_components(cfg=cfg)
    except Exception:
        pass

    cfg = load_from_checkpoint(cfg)

    # 3. CREATE ENVIRONMENT & MODEL
    env = make_env_func_batched(cfg, env_config=None)
    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()
    
    log.info(f"Successfully loaded model from experiment: {cfg.experiment}")

    # 4. LOCATE THE Z-MODULATION WEIGHTS
    mod_type = getattr(cfg, "Decoder_context_mod", "additive")
    target_layer_name = "film_gamma.weight" if mod_type == "FiLM" else "context_proj.weight"
    
    z_weights = None
    for name, param in actor_critic.named_parameters():
        if target_layer_name in name:
            log.info(f"Found Z-Modulation layer: {name} | Shape: {param.shape}")
            z_weights = param.detach()
            break
            
    if z_weights is None:
        log.error(f"Could not find {target_layer_name}! Available parameters:")
        for name, _ in actor_critic.named_parameters():
            print(f" - {name}")
        return

    # Transpose if necessary so each column is a mode
    if z_weights.shape[0] == 4: 
        z_weights = z_weights.T
        
    K = z_weights.shape[1]
    
    print("\n" + "="*50)
    print(f"MODE COLLAPSE ANALYSIS ({mod_type.upper()})")
    print("="*50)
    
    # Check 1: L2 Norms
    norms = torch.norm(z_weights, p=2, dim=0)
    for k in range(K):
        print(f"Mode {k} Weight Magnitude (L2 Norm): {norms[k].item():.4f}")
        
    print("-" * 50)
    
    # Check 2: Cosine Similarity
    for i, j in itertools.combinations(range(K), 2):
        vec_i = z_weights[:, i]
        vec_j = z_weights[:, j]
        
        cos_sim = F.cosine_similarity(vec_i.unsqueeze(0), vec_j.unsqueeze(0), eps=1e-8).item()
        warning = " <--- ⚠️ COLLAPSED!" if cos_sim > 0.90 else ""
        print(f"Similarity between Mode {i} and Mode {j}: {cos_sim:.4f}{warning}")

if __name__ == "__main__":
    check_weight_collapse()