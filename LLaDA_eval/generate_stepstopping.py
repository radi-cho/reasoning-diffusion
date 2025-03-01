import torch
import numpy as np
import torch.nn.functional as F
torch.cuda.empty_cache()

from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

@torch.no_grad()
def generate_with_second_order_convergence(model, prompt, max_steps, min_steps, 
                                          gen_length, block_length, window_size,
                                          convergence_threshold,
                                          remasking, temperature=0., cfg_scale=0.,
                                           mask_id=126336,
                                          ):
    """
    Generate text with LLaDA model using second-order convergence detection for early stopping.
    
    Args:
        model: Mask predictor model
        prompt: Input prompt tensor
        max_steps: Maximum number of sampling steps
        min_steps: Minimum number of sampling steps before early stopping can be considered
        gen_length: Length of generated text
        block_length: Size of generation blocks for semi-autoregressive generation
        temperature: Sampling temperature for Gumbel noise
        cfg_scale: Classifier-free guidance scale
        remasking: Remasking strategy ('low_confidence' or 'random')
        mask_id: Token ID for mask token
        window_size: Window size for convergence detection
        convergence_threshold: Threshold for second derivative to determine convergence
    """
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    # Ensure max_steps is divisible by num_blocks
    if max_steps % num_blocks != 0:
        max_steps = (max_steps // num_blocks) * num_blocks
    
    steps_per_block = max_steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: 
                              prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        
        # Calculate number of tokens to transition at each step for this block
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        
        # List to track confidence history
        confidence_history = []
        
        # Loop for maximum number of steps for this block
        actual_steps = 0
        for i in range(steps_per_block):
            actual_steps += 1
            mask_index = (x == mask_id)
            
            # Apply classifier-free guidance if needed
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            # Apply Gumbel noise for sampling
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            # Calculate confidence scores for remasking
            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            # Prevent remasking beyond current block 
            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -float('inf')

            # Calculate average confidence for masked tokens to track convergence
            masked_confidences = x0_p[mask_index]
            if masked_confidences.numel() > 0:
                avg_confidence = masked_confidences.mean().item()
                confidence_history.append(avg_confidence)
            
                # Apply second-order convergence detection
                if i >= min_steps and len(confidence_history) > window_size + 1:
                    # Check if we've reached convergence
                    if check_second_order_convergence(confidence_history, window_size, convergence_threshold):
                        print(f"Block {num_block+1}/{num_blocks}: Early stopping at step {i+1}/{steps_per_block}")
                        break

            # Update token values
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -float('inf'))

            # Select which tokens to update based on confidence and number to transfer
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
        
        print(f"Block {num_block+1}/{num_blocks} completed in {actual_steps}/{steps_per_block} steps")

    return x, actual_steps 

def check_second_order_convergence(confidence_history, window_size=5, threshold=0.001):
    """
    Check if the rate of improvement in confidence scores is diminishing,
    indicating convergence.
    
    Args:
        confidence_history: List of average confidence scores over time
        window_size: Window size for calculating derivatives
        threshold: Threshold for determining convergence
        
    Returns:
        bool: True if convergence detected, False otherwise
    """
    if len(confidence_history) < window_size + 1:
        return False
    
    # Get the most recent confidence scores
    recent_scores = confidence_history[-window_size-1:]
    
    # Calculate first derivatives (changes in confidence)
    first_derivatives = [recent_scores[i+1] - recent_scores[i] for i in range(len(recent_scores)-1)]
    
    # Calculate second derivatives (changes in the rate of change)
    second_derivatives = [first_derivatives[i+1] - first_derivatives[i] 
                         for i in range(len(first_derivatives)-1)]
    
    # Check if recent second derivatives are negative and small enough
    # This indicates the rate of improvement is diminishing
    convergence_detected = all(sd < 0 for sd in second_derivatives[-3:]) and \
                           all(abs(sd) < threshold for sd in second_derivatives[-3:])
    
    # Also check if first derivatives are small, indicating slow improvement
    slow_improvement = all(abs(fd) < threshold * 5 for fd in first_derivatives[-3:])
    
    return convergence_detected and slow_improvement

