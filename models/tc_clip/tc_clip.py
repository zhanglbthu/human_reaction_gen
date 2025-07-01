"""
TC-CLIP
Copyright (c) 2024-present NAVER Cloud Corp.
CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn

# from .tc_clip_text_encoder import VPTextEncoder
# from .tc_clip_prompt_learner import VPPromptLearner


class TCCLIP_VE(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        # self.prompt_learner = VPPromptLearner(cfg, classnames, clip_model, logger)
        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        # self.text_encoder = VPTextEncoder(clip_model)
        # self.logit_scale = clip_model.logit_scale
        self.dtype = torch.float32 # clip_model.dtype
        # self.prompt_generation_layer_level = self.text_encoder.prompt_generation_layer_level
        # self.return_layer_num = self.prompt_generation_layer_level.copy()
        # if 11 not in self.return_layer_num:
        #     self.return_layer_num.append(11)

        self.return_layer_num = [11]

    def forward(self, image, return_attention=False, return_source=False):
        # Encode visual features
        image_features, context_tokens, attn, source = self.image_encoder(image.type(self.dtype),
                                                                          return_layer_num=self.return_layer_num,
                                                                          return_attention=return_attention,
                                                                          return_source=return_source)

        # Now take the mean along the temporal direction with last layer cls tokens
        image_features_mean = image_features[:, -1, ...].mean(dim=1, keepdim=False)  # [b, 512]
        image_features = image_features.squeeze(1) # [b, frame_num, 512]
        return image_features_mean, image_features
