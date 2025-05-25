# Code from inbarhub/DDPM_inversoin and from google/prompt-to-prompt

from typing import Union, Optional, List
import torch
import numpy as np
from tqdm import tqdm
from utils import get_text_embeddings

def next_step(ldm_model, model_output: Union[torch.FloatTensor, np.ndarray],
              timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
    timestep, next_timestep = min(timestep - ldm_model.model.scheduler.config.num_train_timesteps
                                  // ldm_model.model.scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = ldm_model.model.scheduler.alphas_cumprod[timestep] if timestep >= 0 else ldm_model.model.scheduler.final_alpha_cumprod
    alpha_prod_t_next = ldm_model.model.scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def get_noise_pred(ldm_model, latent, t, text_emb, uncond_emb, cfg_scale):
    noise_pred_uncond, _, _ = ldm_model.unet_forward(
            latent,
            timestep=t,
            encoder_hidden_states=uncond_emb.embedding_hidden_states,
            class_labels=uncond_emb.embedding_class_lables,
            encoder_attention_mask=uncond_emb.boolean_prompt_mask,
        )

    noise_prediction_text, _, _ = ldm_model.unet_forward(
            latent,
            timestep=t,
            encoder_hidden_states=text_emb.embedding_hidden_states,
            class_labels=text_emb.embedding_class_lables,
            encoder_attention_mask=text_emb.boolean_prompt_mask,
        )

    noise_pred = noise_pred_uncond.sample + cfg_scale * (noise_prediction_text.sample - noise_pred_uncond.sample)
    return noise_pred


@torch.no_grad()
def ddim_inversion(ldm_model, w0, prompts, cfg_scale, num_inference_steps, skip):

    _, text_emb, uncond_emb = get_text_embeddings(prompts, [""], ldm_model)

    latent = w0.clone().detach()
    for i in tqdm(range(num_inference_steps)):
        if num_inference_steps - i <= skip:
            break
        t = ldm_model.model.scheduler.timesteps[len(ldm_model.model.scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred(ldm_model, latent, t, text_emb, uncond_emb, cfg_scale)
        latent = next_step(ldm_model, noise_pred, t, latent)
    return latent

def low_rank_approx_4d(M: torch.Tensor, k: int) -> torch.Tensor:
    """
    Truncated SVD along the [T, F] axes for each [B, C].
    M: [B, C, T, F] -> returns [B, C, T, F] low-rank reconstruction
    """
    B, C, T, F = M.shape
    M_flat = M.view(B * C, T, F)
    U, S, Vh = torch.linalg.svd(M_flat, full_matrices=False)  # U: [BC, T, r], S: [BC, r], Vh: [BC, r, F]
    U_k = U[..., :k]              # [BC, T, k]
    S_k = S[..., :k]              # [BC, k]
    Vh_k = Vh[..., :k, :]         # [BC, k, F]
    # Reconstruct: (U_k * S_k) @ Vh_k  → [BC, T, F]
    M_low = (U_k * S_k.unsqueeze(1)) @ Vh_k
    return M_low.view(B, C, T, F)

@torch.no_grad()
def text2image_ldm_stable(ldm_model, prompt: List[str], num_inference_steps: int = 50,
                          guidance_scale: float = 7.5, xt: Optional[torch.FloatTensor] = None, skip: int = 0):
    _, text_emb, uncond_emb = get_text_embeddings(prompt, [""], ldm_model)

    percentile = 0.75   # which quantile to use as the cutoff
    steepness  = 1000.0   # how sharp the gate is around that cutoff

    for t in tqdm(ldm_model.model.scheduler.timesteps[skip:]):
        noise_pred_uncond, _, _ = ldm_model.unet_forward(
                xt,
                timestep=t,
                encoder_hidden_states=uncond_emb.embedding_hidden_states,
                class_labels=uncond_emb.embedding_class_lables,
                encoder_attention_mask=uncond_emb.boolean_prompt_mask,
            )

        noise_prediction_text, _, _ = ldm_model.unet_forward(
                xt,
                timestep=t,
                encoder_hidden_states=text_emb.embedding_hidden_states,
                class_labels=text_emb.embedding_class_lables,
                encoder_attention_mask=text_emb.boolean_prompt_mask,
            )
        noise_diff = noise_prediction_text.sample - noise_pred_uncond.sample
        noise_pred = noise_pred_uncond.sample + guidance_scale * noise_diff

        xt = ldm_model.model.scheduler.step(noise_pred, t, xt, eta=0).prev_sample

    return xt
