import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from diffusers import AudioLDMPipeline, AudioLDM2Pipeline
from transformers import RobertaTokenizer, RobertaTokenizerFast
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
import os
import json
from typing import Any, Dict, List, Optional, Tuple, Union

class PipelineWrapper(torch.nn.Module):
    def __init__(self, model_id: str,
                 device: torch.device,
                 double_precision: bool = False,
                 token: Optional[str] = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_id = model_id
        self.device = device
        self.double_precision = double_precision
        self.token = token

    def get_sigma(self, timestep: int) -> float:
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.model.scheduler.alphas_cumprod - 1)
        return sqrt_recipm1_alphas_cumprod[timestep]

    def load_scheduler(self) -> None:
        pass

    def get_fn_STFT(self) -> torch.nn.Module:
        pass

    def get_sr(self) -> int:
        return 16000

    def vae_encode(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def vae_decode(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def decode_to_mel(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def setup_extra_inputs(self, *args, **kwargs) -> None:
        pass

    def encode_text(self, prompts: List[str], **kwargs
                    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        pass

    def get_variance(self, timestep: torch.Tensor, prev_timestep: torch.Tensor) -> torch.Tensor:
        pass

    def get_alpha_prod_t_prev(self, prev_timestep: torch.Tensor) -> torch.Tensor:
        pass

    def get_noise_shape(self, x0: torch.Tensor, num_steps: int) -> Tuple[int, ...]:
        variance_noise_shape = (num_steps,
                                self.model.unet.config.in_channels,
                                x0.shape[-2],
                                x0.shape[-1])
        return variance_noise_shape

    def sample_xts_from_x0(self, x0: torch.Tensor, num_inference_steps: int = 50) -> torch.Tensor:
        """
        Samples from P(x_1:T|x_0)
        """
        alpha_bar = self.model.scheduler.alphas_cumprod
        sqrt_one_minus_alpha_bar = (1-alpha_bar) ** 0.5

        variance_noise_shape = self.get_noise_shape(x0, num_inference_steps + 1)
        timesteps = self.model.scheduler.timesteps.to(self.device)
        t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
        xts = torch.zeros(variance_noise_shape).to(x0.device)
        xts[0] = x0
        for t in reversed(timesteps):
            idx = num_inference_steps - t_to_idx[int(t)]
            xts[idx] = x0 * (alpha_bar[t] ** 0.5) + torch.randn_like(x0) * sqrt_one_minus_alpha_bar[t]

        return xts

    def get_zs_from_xts(self, xt: torch.Tensor, xtm1: torch.Tensor, noise_pred: torch.Tensor,
                        t: torch.Tensor, eta: float = 0, numerical_fix: bool = True, **kwargs
                        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # pred of x0
        alpha_bar = self.model.scheduler.alphas_cumprod
        if self.model.scheduler.config.prediction_type == 'epsilon':
            pred_original_sample = (xt - (1 - alpha_bar[t]) ** 0.5 * noise_pred) / alpha_bar[t] ** 0.5
        elif self.model.scheduler.config.prediction_type == 'v_prediction':
            pred_original_sample = (alpha_bar[t] ** 0.5) * xt - ((1 - alpha_bar[t]) ** 0.5) * noise_pred

        # direction to xt
        prev_timestep = t - self.model.scheduler.config.num_train_timesteps // \
            self.model.scheduler.num_inference_steps

        alpha_prod_t_prev = self.get_alpha_prod_t_prev(prev_timestep)
        variance = self.get_variance(t, prev_timestep)

        if self.model.scheduler.config.prediction_type == 'epsilon':
            radom_noise_pred = noise_pred
        elif self.model.scheduler.config.prediction_type == 'v_prediction':
            radom_noise_pred = (alpha_bar[t] ** 0.5) * noise_pred + ((1 - alpha_bar[t]) ** 0.5) * xt

        pred_sample_direction = (1 - alpha_prod_t_prev - eta * variance) ** (0.5) * radom_noise_pred

        mu_xt = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        z = (xtm1 - mu_xt) / (eta * variance ** 0.5)

        # correction to avoid error accumulation
        if numerical_fix:
            xtm1 = mu_xt + (eta * variance ** 0.5)*z

        return z, xtm1, None

    def reverse_step_with_custom_noise(self, model_output: torch.Tensor, timestep: torch.Tensor, sample: torch.Tensor,
                                       variance_noise: Optional[torch.Tensor] = None, eta: float = 0, **kwargs
                                       ) -> torch.Tensor:
        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.model.scheduler.config.num_train_timesteps // \
            self.model.scheduler.num_inference_steps
        # 2. compute alphas, betas
        alpha_prod_t = self.model.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.get_alpha_prod_t_prev(prev_timestep)
        beta_prod_t = 1 - alpha_prod_t
        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.model.scheduler.config.prediction_type == 'epsilon':
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.model.scheduler.config.prediction_type == 'v_prediction':
            pred_original_sample = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_output

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        # variance = self.scheduler._get_variance(timestep, prev_timestep)
        variance = self.get_variance(timestep, prev_timestep)
        # std_dev_t = eta * variance ** (0.5)
        # Take care of asymetric reverse process (asyrp)
        if self.model.scheduler.config.prediction_type == 'epsilon':
            model_output_direction = model_output
        elif self.model.scheduler.config.prediction_type == 'v_prediction':
            model_output_direction = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        # pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output_direction
        pred_sample_direction = (1 - alpha_prod_t_prev - eta * variance) ** (0.5) * model_output_direction
        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        # 8. Add noice if eta > 0
        if eta > 0:
            if variance_noise is None:
                variance_noise = torch.randn(model_output.shape, device=self.device)
            sigma_z = eta * variance ** (0.5) * variance_noise
            prev_sample = prev_sample + sigma_z

        return prev_sample

    def unet_forward(self,
                     sample: torch.FloatTensor,
                     timestep: Union[torch.Tensor, float, int],
                     encoder_hidden_states: torch.Tensor,
                     class_labels: Optional[torch.Tensor] = None,
                     timestep_cond: Optional[torch.Tensor] = None,
                     attention_mask: Optional[torch.Tensor] = None,
                     cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                     added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
                     down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
                     mid_block_additional_residual: Optional[torch.Tensor] = None,
                     encoder_attention_mask: Optional[torch.Tensor] = None,
                     replace_h_space: Optional[torch.Tensor] = None,
                     replace_skip_conns: Optional[Dict[int, torch.Tensor]] = None,
                     return_dict: bool = True,
                     zero_out_resconns: Optional[Union[int, List]] = None) -> Tuple:

        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.model.unet.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            # logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.model.unet.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.model.unet.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.model.unet.time_embedding(t_emb, timestep_cond)

        if self.model.unet.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.model.unet.config.class_embed_type == "timestep":
                class_labels = self.model.unet.time_proj(class_labels)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # there might be better ways to encapsulate this.
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.model.unet.class_embedding(class_labels).to(dtype=sample.dtype)

            if self.model.unet.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        if self.model.unet.config.addition_embed_type == "text":
            aug_emb = self.model.unet.add_embedding(encoder_hidden_states)
            emb = emb + aug_emb
        elif self.model.unet.config.addition_embed_type == "text_image":
            # Kadinsky 2.1 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.model.unet.__class__} has the config param `addition_embed_type` set to 'text_image' "
                    f"which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )

            image_embs = added_cond_kwargs.get("image_embeds")
            text_embs = added_cond_kwargs.get("text_embeds", encoder_hidden_states)

            aug_emb = self.model.unet.add_embedding(text_embs, image_embs)
            emb = emb + aug_emb

        if self.model.unet.time_embed_act is not None:
            emb = self.model.unet.time_embed_act(emb)

        if self.model.unet.encoder_hid_proj is not None and self.model.unet.config.encoder_hid_dim_type == "text_proj":
            encoder_hidden_states = self.model.unet.encoder_hid_proj(encoder_hidden_states)
        elif self.model.unet.encoder_hid_proj is not None and \
                self.model.unet.config.encoder_hid_dim_type == "text_image_proj":
            # Kadinsky 2.1 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.model.unet.__class__} has the config param `encoder_hid_dim_type` set to 'text_image_proj' "
                    f"which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )

            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.model.unet.encoder_hid_proj(encoder_hidden_states, image_embeds)

        # 2. pre-process
        sample = self.model.unet.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.model.unet.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.model.unet.mid_block is not None:
            sample = self.model.unet.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            )

        # print(sample.shape)

        if replace_h_space is None:
            h_space = sample.clone()
        else:
            h_space = replace_h_space
            sample = replace_h_space.clone()

        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        extracted_res_conns = {}
        # 5. up
        for i, upsample_block in enumerate(self.model.unet.up_blocks):
            is_final_block = i == len(self.model.unet.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            if replace_skip_conns is not None and replace_skip_conns.get(i):
                res_samples = replace_skip_conns.get(i)

            if zero_out_resconns is not None:
                if (type(zero_out_resconns) is int and i >= (zero_out_resconns - 1)) or \
                        type(zero_out_resconns) is list and i in zero_out_resconns:
                    res_samples = [torch.zeros_like(x) for x in res_samples]
                # down_block_res_samples = [torch.zeros_like(x) for x in down_block_res_samples]

            extracted_res_conns[i] = res_samples

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )

        # 6. post-process
        if self.model.unet.conv_norm_out:
            sample = self.model.unet.conv_norm_out(sample)
            sample = self.model.unet.conv_act(sample)
        sample = self.model.unet.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample), h_space, extracted_res_conns


class TangoWrapper(PipelineWrapper):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from tango.audioldm.audio.stft import TacotronSTFT as TangoTacotronSTFT
        from tango.audioldm.variational_autoencoder import AutoencoderKL as TangoAutoencoderKL
        from tango.models import AudioDiffusion as TangoAudioDiffusion
        from huggingface_hub import snapshot_download

        path = snapshot_download(repo_id=self.model_id, token=self.token)
        vae_config = json.load(open("{}/vae_config.json".format(path)))
        stft_config = json.load(open("{}/stft_config.json".format(path)))
        main_config = json.load(open("{}/main_config.json".format(path)))
        main_config['unet_model_config_path'] = os.path.join('tango', main_config['unet_model_config_path'])

        self.vae = TangoAutoencoderKL(**vae_config).to(self.device)
        self.stft = TangoTacotronSTFT(**stft_config)
        self.model = TangoAudioDiffusion(**main_config).to(self.device)

        vae_weights = torch.load("{}/pytorch_model_vae.bin".format(path), map_location=self.device)
        stft_weights = torch.load("{}/pytorch_model_stft.bin".format(path), map_location=torch.device('cpu'))
        torch.cuda.empty_cache()

        main_weights = torch.load("{}/pytorch_model_main.bin".format(path), map_location=self.device)

        self.vae.load_state_dict(vae_weights)
        self.stft.load_state_dict(stft_weights)
        self.model.load_state_dict(main_weights)

        self.vae.eval()
        self.stft.eval()
        self.model.eval()

        del vae_weights, stft_weights, main_weights
        torch.cuda.empty_cache()

        self.scheduler_name = main_config["scheduler_name"]

    def load_scheduler(self) -> None:
        self.model.scheduler = DDIMScheduler.from_pretrained(self.scheduler_name, subfolder="scheduler")

    def get_fn_STFT(self) -> torch.nn.Module:
        return self.stft

    def vae_encode(self, x: torch.Tensor) -> torch.Tensor:
        # unwrapped_vae = accelerator.unwrap_model(vae)
        if x.shape[2] % 4:
            x = torch.nn.functional.pad(x, (0, 0, 4 - (x.shape[2] % 4), 0))

        if x.shape[2] > 1700:
            raise RuntimeWarning("This model dies at this point")

        return self.vae.get_first_stage_encoding(self.vae.encode_first_stage(x))

    def vae_decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.vae.decode_first_stage(x)

    def decode_to_mel(self, x: torch.Tensor) -> torch.Tensor:
        return self.vae.decode_to_waveform(x)

    def encode_text(self, prompts: List[str], **kwargs) -> Tuple[Optional[torch.Tensor], None, Optional[torch.Tensor]]:
        prompt_embeds, boolean_prompt_mask = self.model.encode_text(prompts)
        # prompt_embeds = F.normalize(prompt_embeds, dim=-1)
        prompt_embeds = prompt_embeds.repeat_interleave(1, 0)
        boolean_prompt_mask = boolean_prompt_mask.repeat_interleave(1, 0)
        return prompt_embeds, None, boolean_prompt_mask

    def get_variance(self, timestep: torch.Tensor, prev_timestep: torch.Tensor) -> torch.Tensor:
        alpha_prod_t = self.model.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.get_alpha_prod_t_prev(prev_timestep)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance

    def get_alpha_prod_t_prev(self, prev_timestep: torch.Tensor) -> torch.Tensor:
        return self.model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 \
            else self.model.scheduler.final_alpha_cumprod


class AudioLDMWrapper(PipelineWrapper):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = AudioLDMPipeline.from_pretrained(self.model_id, token=self.token).to(self.device)

    def load_scheduler(self) -> None:
        self.model.scheduler = DDIMScheduler.from_pretrained(self.model_id, subfolder="scheduler")

    def get_fn_STFT(self) -> torch.nn.Module:
        from audioldm.audio import TacotronSTFT
        return TacotronSTFT(
            filter_length=1024,
            hop_length=160,
            win_length=1024,
            n_mel_channels=64,
            sampling_rate=16000,
            mel_fmin=0,
            mel_fmax=8000,
        )

    def vae_encode(self, x: torch.Tensor) -> torch.Tensor:
        # self.model.vae.disable_tiling()
        if x.shape[2] % 4:
            x = torch.nn.functional.pad(x, (0, 0, 4 - (x.shape[2] % 4), 0))
        return (self.model.vae.encode(x).latent_dist.mode() * self.model.vae.config.scaling_factor).float()
        # return (self.encode_no_tiling(x).latent_dist.mode() * self.model.vae.config.scaling_factor).float()

    def vae_decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.vae.decode(1 / self.model.vae.config.scaling_factor * x).sample

    def decode_to_mel(self, x: torch.Tensor) -> torch.Tensor:
        if self.double_precision:
            return self.model.mel_spectrogram_to_waveform(x[0, 0].detach().double()).detach().unsqueeze(0)

        return self.model.mel_spectrogram_to_waveform(x[0, 0].detach().float()).detach().unsqueeze(0)

    def encode_text(self, prompts: List[str], **kwargs) -> Tuple[None, Optional[torch.Tensor], None]:
        text_input = self.model.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_input.input_ids
        attention_mask = text_input.attention_mask
        untruncated_ids = self.model.tokenizer(prompts, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] \
                and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.model.tokenizer.batch_decode(
                untruncated_ids[:, self.model.tokenizer.model_max_length - 1: -1])
            print("The following part of your input was truncated because CLAP can only handle sequences up to"
                  f" {self.model.tokenizer.model_max_length} tokens: {removed_text}")

        with torch.no_grad():
            text_encoding = self.model.text_encoder(text_input.input_ids.to(self.device),
                                                    attention_mask=attention_mask.to(self.device))[0]
        text_encoding = F.normalize(text_encoding, dim=-1)
        text_encoding = text_encoding.to(dtype=self.model.text_encoder.dtype, device=self.device)

        return None, text_encoding, None

    def get_variance(self, timestep: torch.Tensor, prev_timestep: torch.Tensor) -> torch.Tensor:
        alpha_prod_t = self.model.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.get_alpha_prod_t_prev(prev_timestep)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance

    def get_alpha_prod_t_prev(self, prev_timestep: torch.Tensor) -> torch.Tensor:
        return self.model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 \
            else self.model.scheduler.final_alpha_cumprod


class AudioLDM2Wrapper(PipelineWrapper):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.double_precision:
            self.model = AudioLDM2Pipeline.from_pretrained(self.model_id, torch_dtype=torch.float64, token=self.token
                                                           ).to(self.device)
        else:
            try:
                self.model = AudioLDM2Pipeline.from_pretrained(self.model_id, local_files_only=False, token=self.token
                                                               ).to(self.device)
            except FileNotFoundError:
                self.model = AudioLDM2Pipeline.from_pretrained(self.model_id, local_files_only=False, token=self.token
                                                               ).to(self.device)

    def load_scheduler(self) -> None:
        self.model.scheduler = DDIMScheduler.from_pretrained(self.model_id, subfolder="scheduler")

    def get_fn_STFT(self) -> torch.nn.Module:
        from audioldm.audio import TacotronSTFT
        return TacotronSTFT(
            filter_length=1024,
            hop_length=160,
            win_length=1024,
            n_mel_channels=64,
            sampling_rate=16000,
            mel_fmin=0,
            mel_fmax=8000,
        )

    def vae_encode(self, x: torch.Tensor) -> torch.Tensor:
        # self.model.vae.disable_tiling()
        if x.shape[2] % 4:
            x = torch.nn.functional.pad(x, (0, 0, 4 - (x.shape[2] % 4), 0))
        return (self.model.vae.encode(x).latent_dist.mode() * self.model.vae.config.scaling_factor).float()
        # return (self.encode_no_tiling(x).latent_dist.mode() * self.model.vae.config.scaling_factor).float()

    def vae_decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.vae.decode(1 / self.model.vae.config.scaling_factor * x).sample

    def decode_to_mel(self, x: torch.Tensor) -> torch.Tensor:
        if self.double_precision:
            tmp = self.model.mel_spectrogram_to_waveform(x[:, 0].detach().double()).detach()
        tmp = self.model.mel_spectrogram_to_waveform(x[:, 0].detach().float()).detach()
        if len(tmp.shape) == 1:
            tmp = tmp.unsqueeze(0)
        return tmp

    def encode_text(self, prompts: List[str], **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokenizers = [self.model.tokenizer, self.model.tokenizer_2]
        text_encoders = [self.model.text_encoder, self.model.text_encoder_2]
        prompt_embeds_list = []
        attention_mask_list = []

        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                prompts,
                padding="max_length" if isinstance(tokenizer, (RobertaTokenizer, RobertaTokenizerFast)) else True,
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids
            attention_mask = text_inputs.attention_mask
            untruncated_ids = tokenizer(prompts, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] \
                    and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = tokenizer.batch_decode(
                    untruncated_ids[:, tokenizer.model_max_length - 1: -1])
                print(f"The following part of your input was truncated because {text_encoder.config.model_type} can "
                      f"only handle sequences up to {tokenizer.model_max_length} tokens: {removed_text}"
                      )

            text_input_ids = text_input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            with torch.no_grad():
                if text_encoder.config.model_type == "clap":
                    prompt_embeds = text_encoder.get_text_features(
                        text_input_ids,
                        attention_mask=attention_mask,
                    )
                    # append the seq-len dim: (bs, hidden_size) -> (bs, seq_len, hidden_size)
                    prompt_embeds = prompt_embeds[:, None, :]
                    # make sure that we attend to this single hidden-state
                    attention_mask = attention_mask.new_ones((len(prompts), 1))
                else:
                    prompt_embeds = text_encoder(
                        text_input_ids,
                        attention_mask=attention_mask,
                    )
                    prompt_embeds = prompt_embeds[0]

            prompt_embeds_list.append(prompt_embeds)
            attention_mask_list.append(attention_mask)

        # print(f'prompt[0].shape: {prompt_embeds_list[0].shape}')
        # print(f'prompt[1].shape: {prompt_embeds_list[1].shape}')
        # print(f'attn[0].shape: {attention_mask_list[0].shape}')
        # print(f'attn[1].shape: {attention_mask_list[1].shape}')

        projection_output = self.model.projection_model(
            hidden_states=prompt_embeds_list[0],
            hidden_states_1=prompt_embeds_list[1],
            attention_mask=attention_mask_list[0],
            attention_mask_1=attention_mask_list[1],
        )
        projected_prompt_embeds = projection_output.hidden_states
        projected_attention_mask = projection_output.attention_mask

        generated_prompt_embeds = self.model.generate_language_model(
            projected_prompt_embeds,
            attention_mask=projected_attention_mask,
            max_new_tokens=None,
        )

        prompt_embeds = prompt_embeds.to(dtype=self.model.text_encoder_2.dtype, device=self.device)
        attention_mask = (
            attention_mask.to(device=self.device)
            if attention_mask is not None
            else torch.ones(prompt_embeds.shape[:2], dtype=torch.long, device=self.device)
        )
        generated_prompt_embeds = generated_prompt_embeds.to(dtype=self.model.language_model.dtype, device=self.device)

        return generated_prompt_embeds, prompt_embeds, attention_mask

    def get_variance(self, timestep: torch.Tensor, prev_timestep: torch.Tensor) -> torch.Tensor:
        alpha_prod_t = self.model.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.get_alpha_prod_t_prev(prev_timestep)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance

    def get_alpha_prod_t_prev(self, prev_timestep: torch.Tensor) -> torch.Tensor:
        return self.model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 \
            else self.model.scheduler.final_alpha_cumprod

    def unet_forward(self,
                     sample: torch.FloatTensor,
                     timestep: Union[torch.Tensor, float, int],
                     encoder_hidden_states: torch.Tensor,
                     timestep_cond: Optional[torch.Tensor] = None,
                     class_labels: Optional[torch.Tensor] = None,
                     attention_mask: Optional[torch.Tensor] = None,
                     encoder_attention_mask: Optional[torch.Tensor] = None,
                     return_dict: bool = True,
                     cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                     mid_block_additional_residual: Optional[torch.Tensor] = None,
                     replace_h_space: Optional[torch.Tensor] = None,
                     replace_skip_conns: Optional[Dict[int, torch.Tensor]] = None,
                     zero_out_resconns: Optional[Union[int, List]] = None) -> Tuple:

        # Translation
        encoder_hidden_states_1 = class_labels
        class_labels = None
        encoder_attention_mask_1 = encoder_attention_mask
        encoder_attention_mask = None

        # return self.model.unet(sample, timestep,
        #                        encoder_hidden_states=generated_prompt_embeds,
        #                        encoder_hidden_states_1=encoder_hidden_states_1,
        #                        encoder_attention_mask_1=encoder_attention_mask_1,
        #                        ), None, None

        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2 ** self.model.unet.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            # print("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        if encoder_attention_mask_1 is not None:
            encoder_attention_mask_1 = (1 - encoder_attention_mask_1.to(sample.dtype)) * -10000.0
            encoder_attention_mask_1 = encoder_attention_mask_1.unsqueeze(1)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.model.unet.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.model.unet.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        if self.model.unet.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.model.unet.config.class_embed_type == "timestep":
                class_labels = self.model.unet.time_proj(class_labels)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # there might be better ways to encapsulate this.
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.model.unet.class_embedding(class_labels).to(dtype=sample.dtype)

            if self.model.unet.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.model.unet.time_embed_act is not None:
            emb = self.model.unet.time_embed_act(emb)

        # 2. pre-process
        sample = self.model.unet.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.model.unet.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    encoder_hidden_states_1=encoder_hidden_states_1,
                    encoder_attention_mask_1=encoder_attention_mask_1,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        if self.model.unet.mid_block is not None:
            sample = self.model.unet.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
                encoder_hidden_states_1=encoder_hidden_states_1,
                encoder_attention_mask_1=encoder_attention_mask_1,
            )

        if replace_h_space is None:
            h_space = sample.clone()
        else:
            h_space = replace_h_space
            sample = replace_h_space.clone()

        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        extracted_res_conns = {}
        # 5. up
        for i, upsample_block in enumerate(self.model.unet.up_blocks):
            is_final_block = i == len(self.model.unet.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            if replace_skip_conns is not None and replace_skip_conns.get(i):
                res_samples = replace_skip_conns.get(i)

            if zero_out_resconns is not None:
                if (type(zero_out_resconns) is int and i >= (zero_out_resconns - 1)) or \
                        type(zero_out_resconns) is list and i in zero_out_resconns:
                    res_samples = [torch.zeros_like(x) for x in res_samples]
                # down_block_res_samples = [torch.zeros_like(x) for x in down_block_res_samples]

            extracted_res_conns[i] = res_samples

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    encoder_hidden_states_1=encoder_hidden_states_1,
                    encoder_attention_mask_1=encoder_attention_mask_1,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )

        # 6. post-process
        if self.model.unet.conv_norm_out:
            sample = self.model.unet.conv_norm_out(sample)
            sample = self.model.unet.conv_act(sample)
        sample = self.model.unet.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample), h_space, extracted_res_conns


def load_model(model_id: str, device: torch.device, num_diffusion_steps: int,
               double_precision: bool = False, token: Optional[str] = None) -> PipelineWrapper:
    if 'tango' in model_id:
        ldm_stable = TangoWrapper(model_id=model_id, device=device, double_precision=double_precision, token=token)
    elif 'audioldm2' in model_id:
        ldm_stable = AudioLDM2Wrapper(model_id=model_id, device=device, double_precision=double_precision, token=token)
    elif 'audioldm' in model_id:
        ldm_stable = AudioLDMWrapper(model_id=model_id, device=device, double_precision=double_precision, token=token)
    ldm_stable.load_scheduler()
    ldm_stable.model.scheduler.set_timesteps(num_diffusion_steps, device=device)
    torch.cuda.empty_cache()
    return ldm_stable
