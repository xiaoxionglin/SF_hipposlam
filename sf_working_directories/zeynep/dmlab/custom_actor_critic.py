from typing import Dict
from sample_factory.model.actor_critic import default_make_actor_critic_func, ActorCritic
import torch

def add_custom_summaries(actor_critic: ActorCritic) -> ActorCritic:
    """Dynamically extends the summaries method to pull custom stats from the Core."""
    original_summaries = actor_critic.summaries
    
    def custom_summaries() -> Dict:
        s = original_summaries()
        
        # Locate the core
        core = getattr(actor_critic, 'core', getattr(actor_critic, 'actor_core', None))
            
        # Extract our variables!
        if core is not None and hasattr(core, 'last_v_L'):
            s['context_rnn/v_L'] = core.last_v_L
            s['context_rnn/v_R'] = core.last_v_R
            s['context_rnn/v_diff'] = core.last_v_diff
            
            # Log the learned decay parameter if it exists
            if hasattr(core, 'context_rnn'):
                if hasattr(core.context_rnn, 'alpha'):
                    s['context_rnn/alpha'] = float(core.context_rnn.alpha.mean().item())
                elif hasattr(core.context_rnn, 'current_alpha'):
                    s['context_rnn/current_alpha'] = float(core.context_rnn.current_alpha.mean().item())
        return s
        
    actor_critic.summaries = custom_summaries
    return actor_critic

def HighLevel_LossWrapper(actor_critic: ActorCritic) -> ActorCritic:
    """Dynamically extends the forward method to include Q-learning or policy loss from the high-level RNN."""
    original_forward = actor_critic.forward
    original_summaries = actor_critic.summaries
    
    def custom_forward(*args, **kwargs):
        # 1. Call the original forward function
        # This returns the TensorDict containing actions, values, and new_rnn_states
        result_dict = original_forward(*args, **kwargs)
        
        # 2. Locate the core
        core = getattr(actor_critic, 'core', getattr(actor_critic, 'actor_core', None))
        
        # 3. Extract the Q-loss if it exists
        if core is not None and hasattr(core, 'last_hl_loss'):
            result_dict['hl_loss'] = core.last_hl_loss
        else:
            # Fallback for inference pass to prevent crashes
            # (Grabs the device from the 'values' tensor safely)
            device = result_dict['values'].device if 'values' in result_dict else torch.device('cpu')
            result_dict['hl_loss'] = torch.tensor(0.0, device=device)
        
        return result_dict
    
    def summaries_with_hl_loss() -> Dict:
        s = original_summaries()
        core = getattr(actor_critic, 'core', getattr(actor_critic, 'actor_core', None))

        if core is not None:
            # Add high-level loss to the summaries
            if hasattr(core, 'last_hl_loss') and core.last_hl_loss is not None:
                s['hl/hl_loss'] = core.last_hl_loss.item() # Use .item() for logging!

            if hasattr(core, 'last_hl_metrics') and core.last_hl_metrics is not None:
                for key, value in core.last_hl_metrics.items():
                    s[key] = value
                
            if hasattr(core, 'last_log_dict') and core.last_log_dict is not None:
                for key, value in core.last_log_dict.items():
                    if isinstance(value, torch.Tensor):
                        s[f'hl/{key}'] = value.float().mean().item()
                    else:
                        s[f'hl/{key}'] = value
        return s
    
    # Override the forward method
    actor_critic.forward = custom_forward
    actor_critic.summaries = summaries_with_hl_loss
    return actor_critic


def make_hipposlam_actor_critic(cfg, obs_space, action_space) -> ActorCritic:
    # Use Sample Factory's default creation logic
    actor_critic = default_make_actor_critic_func(cfg, obs_space, action_space)
    return HighLevel_LossWrapper(actor_critic)
    #return add_custom_summaries(actor_critic)  