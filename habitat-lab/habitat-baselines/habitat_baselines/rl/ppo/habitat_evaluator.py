import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
import tqdm

from habitat import logger
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
)
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.rl.ppo.evaluator import Evaluator, pause_envs
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_action_space_info,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.info_dict import extract_scalars_from_info
from corruptions.my_benchmark import MyBenchmark
from corruptions.albumentations_benchmark import AlbumentationsBenchmark
import cv2
from .corruption_types import robustnav_corruption_types, albumentations_corrupton_types
from .ae.apply_recon import apply_ae
class HabitatEvaluator(Evaluator):
    """
    Evaluator for Habitat environments.
    """

    def evaluate_agent(
        self,
        agent,
        envs,
        config,
        checkpoint_index,
        step_id,
        writer,
        device,
        obs_transforms,
        env_spec,
        rank0_keys,
        mom_pre, # DAU adaptation
        decay_factor, # DUA adaptation
        min_mom, # DUA adaptation
    ):
        self.count = 0
        self.my_benchmark = None
        apply_corruptions = config.robustness.apply_corruptions
        try: # Each observation comes from the decoder output instead
            apply_recon = config.recon.apply_recon
        except:
            apply_recon = False
            
        try:
            aae_path = config.recon.aae_path
        except:
            aae_path = None


        self.apply_ablation = False #TODO config.ablation.run
        self.ablation_block = None #TODO config.ablation.block


        # self.apply_ablation = False
        # self.ablation_block = None
        if apply_recon:
            print(f"Apply Recon {apply_recon}")
            # print(f"Apply Ablation {self.apply_ablation} on block {self.ablation_block}")
            adapt = config.recon.adapt_encoder
            self.ae_recon = apply_ae(device, adapt, self.apply_ablation, self.ablation_block, aae_path)


        if apply_corruptions:
            if config.robustness.visual_corruption in robustnav_corruption_types:
                print(f"use robustnav: {config.robustness.visual_corruption}")
                self.my_benchmark = MyBenchmark(config)
            elif config.robustness.visual_corruption in albumentations_corrupton_types:
                print(f"use albumentations: {config.robustness.visual_corruption}")
                self.my_benchmark = AlbumentationsBenchmark(config)

        observations = envs.reset()
        observations = envs.post_step(observations)
        if apply_corruptions:
            for i in range(len(observations)):
                temp = self.my_benchmark.corrupt_rgb_observation(observations[i]["rgb"])
                print("real", observations[i]["rgb"].shape)
                print("corrupted", temp.shape)
                if len(temp.shape) == len(observations[i]["rgb"].shape): # For motion blur
                    if apply_recon:
                        temp = self.ae_recon.recon(temp)
                        print("Apply Recon First Step Successfully")
                    observations[i]["rgb"] = temp
                else:
                    print("Motion blur images error with shape", temp.shape)
        if "vit" in config.habitat_baselines.rl.ddppo.backbone: # Check image size for ViT
            observations = check_size_for_vit(observations, 224)
            
        batch = batch_obs(observations, device=device)
        batch = apply_obs_transforms_batch(batch, obs_transforms)  # type: ignore

        action_shape, discrete_actions = get_action_space_info(
            agent.actor_critic.policy_action_space
        )

        current_episode_reward = torch.zeros(envs.num_envs, 1, device="cpu")

        test_recurrent_hidden_states = torch.zeros(
            (
                config.habitat_baselines.num_environments,
                *agent.actor_critic.hidden_state_shape,
            ),
            device=device,
        )

        hidden_state_lens = agent.actor_critic.hidden_state_shape_lens
        action_space_lens = agent.actor_critic.policy_action_space_shape_lens

        prev_actions = torch.zeros(
            config.habitat_baselines.num_environments,
            *action_shape,
            device=device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            config.habitat_baselines.num_environments,
            *agent.masks_shape,
            device=device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)

        if len(config.habitat_baselines.eval.video_option) > 0:
            # Add the first frame of the episode to the video.
            rgb_frames: List[List[np.ndarray]] = [
                [
                    observations_to_image(
                        {k: v[env_idx] for k, v in batch.items()}, {}
                    )
                ]
                for env_idx in range(config.habitat_baselines.num_environments)
            ]
        else:
            rgb_frames = None

        if len(config.habitat_baselines.eval.video_option) > 0:
            os.makedirs(config.habitat_baselines.video_dir, exist_ok=True)

        number_of_eval_episodes = config.habitat_baselines.test_episode_count
        evals_per_ep = config.habitat_baselines.eval.evals_per_ep
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(envs.number_of_episodes)
            print(f"NUM EVAL EPISODES= {number_of_eval_episodes}")
        else:
            total_num_eps = sum(envs.number_of_episodes)
            # if total_num_eps is negative, it means the number of evaluation episodes is unknown
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert (
            number_of_eval_episodes > 0
        ), "You must specify a number of evaluation episodes with test_episode_count"

        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)
        agent.eval()

        while (
            len(stats_episodes) < (number_of_eval_episodes * evals_per_ep)
            and envs.num_envs > 0
        ):
            space_lengths = {}
            n_agents = len(config.habitat.simulator.agents)
            # Ablation of DUA
            # if not(self.apply_ablation): # Normal DUA run
            # else: # Ablation DUA on specific block (self.ablation_block)
            #     assert n_agents == 1
            #     agent, mom_pre, decay_factor, min_mom = self._adapt_ablation(agent, mom_pre, decay_factor, min_mom)                
                # if len(stats_episodes):
                #     print(f"apply abaltion on:{self.ablation_block}")


            current_episodes_info = envs.current_episodes()


            if n_agents > 1:
                space_lengths = {
                    "index_len_recurrent_hidden_states": hidden_state_lens,
                    "index_len_prev_actions": action_space_lens,
                }
            with inference_mode():
                # Use DUA
                if config.adaptation.adaptation_phase and config.adaptation.adaptation_method == "dua":
                    assert n_agents == 1
                    agent, mom_pre, decay_factor, min_mom = self._adapt(agent, 
                                                                        mom_pre, 
                                                                        decay_factor, 
                                                                        min_mom) # BatchNorm.train() to update running statistics
                    _ = agent.actor_critic.act(
                                                batch,
                                                test_recurrent_hidden_states,
                                                prev_actions,
                                                not_done_masks,
                                                deterministic=False,
                                                **space_lengths,
                                            ) # dummy feedforward to pdate running statistics
                    agent.eval() # BatchNorm.eval() to use running statistics in normalization
                    if self.count == 0:
                        print("Apply DUA")

                action_data = agent.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                    **space_lengths,
                )
                if action_data.should_inserts is None:
                    test_recurrent_hidden_states = (
                        action_data.rnn_hidden_states
                    )
                    prev_actions.copy_(action_data.actions)  # type: ignore
                else:
                    agent.actor_critic.update_hidden_state(
                        test_recurrent_hidden_states, prev_actions, action_data
                    )

            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            if is_continuous_action_space(env_spec.action_space):
                # Clipping actions to the specified limits
                step_data = [
                    np.clip(
                        a.numpy(),
                        env_spec.action_space.low,
                        env_spec.action_space.high,
                    )
                    for a in action_data.env_actions.cpu()
                ]
            else:
                step_data = [a.item() for a in action_data.env_actions.cpu()]

            outputs = envs.step(step_data)

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            # Note that `policy_infos` represents the information about the
            # action BEFORE `observations` (the action used to transition to
            # `observations`).
            policy_infos = agent.actor_critic.get_extra(
                action_data, infos, dones
            )
            for i in range(len(policy_infos)):
                infos[i].update(policy_infos[i])

            observations = envs.post_step(observations)
            if apply_corruptions:
                for i in range(len(observations)):
                    temp = self.my_benchmark.corrupt_rgb_observation(observations[i]["rgb"])
                    if len(temp.shape) == len(observations[i]["rgb"].shape): # For motion blur
                        if apply_recon:
                            temp = self.ae_recon.recon(temp)
                        observations[i]["rgb"] = temp
                    else:
                        print("Motion blur images error with shape", temp.shape)

            else:
                if apply_recon:
                    for i in range(len(observations)):
                        temp = self.ae_recon.recon(observations[i]["rgb"])
                        observations[i]["rgb"] = temp
                        

            if "vit" in config.habitat_baselines.rl.ddppo.backbone: # Check image size for ViT
                observations = check_size_for_vit(observations, 224)
            batch = batch_obs(  # type: ignore
                observations,
                device=device,
            )
            batch = apply_obs_transforms_batch(batch, obs_transforms)  # type: ignore

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            ).repeat(1, *agent.masks_shape)

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes_info = envs.current_episodes()
            envs_to_pause = []
            n_envs = envs.num_envs
            for i in range(n_envs):
                if (
                    ep_eval_count[
                        (
                            next_episodes_info[i].scene_id,
                            next_episodes_info[i].episode_id,
                        )
                    ]
                    == evals_per_ep
                ):
                    envs_to_pause.append(i)

                # Exclude the keys from `_rank0_keys` from displaying in the video
                disp_info = {
                    k: v for k, v in infos[i].items() if k not in rank0_keys
                }

                if len(config.habitat_baselines.eval.video_option) > 0:
                    # TODO move normalization / channel changing out of the policy and undo it here
                    frame = observations_to_image(
                        {k: v[i] for k, v in batch.items()}, disp_info
                    )
                    if not not_done_masks[i].any().item():
                        # The last frame corresponds to the first frame of the next episode
                        # but the info is correct. So we use a black frame
                        final_frame = observations_to_image(
                            {k: v[i] * 0.0 for k, v in batch.items()},
                            disp_info,
                        )
                        final_frame = overlay_frame(final_frame, disp_info)
                        rgb_frames[i].append(final_frame)
                        # The starting frame of the next episode will be the final element..
                        rgb_frames[i].append(frame)
                    else:
                        frame = overlay_frame(frame, disp_info)
                        rgb_frames[i].append(frame)
                    
                # Collect Dataset: Temporary Code for crearing an image dataset)
                if self.count < 200:
                    frame = observations_to_image(
                        {k: v[i] for k, v in batch.items()}, disp_info
                    )
                    os.makedirs(config.habitat_baselines.video_dir, exist_ok=True)
                    base_episode = current_episodes_info[0]
                    scene_id = base_episode.scene_id
                    scene_id = os.path.basename(scene_id)
                    episode_id = base_episode.episode_id
                    img_save_dir = config.habitat_baselines.video_dir
                    image_save_path = f'{img_save_dir}/{scene_id}_ep={episode_id}_frameid={self.count}.png'
                    frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(image_save_path, frame_bgr)
                    if self.count == 0:
                        print("Frame shape", frame.shape)
                        print(f"saved at {image_save_path}")
                    self.count += 1
    

                # episode ended
                if not not_done_masks[i].any().item():
                    base_episode = current_episodes_info[0]
                    episode_id = base_episode.episode_id
                    extracted_infos = extract_scalars_from_info(infos[i])
                    for k, v in extracted_infos.items():
                        writer.add_scalar(f"eval_by_episode_metrics/{k}", v, len(stats_episodes))

                    pbar.update()
                    episode_stats = {
                        "reward": current_episode_reward[i].item()
                    }
                    episode_stats.update(extract_scalars_from_info(infos[i]))
                    current_episode_reward[i] = 0
                    k = (
                        current_episodes_info[i].scene_id,
                        current_episodes_info[i].episode_id,
                    )
                    ep_eval_count[k] += 1
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats

                    if len(config.habitat_baselines.eval.video_option) > 0:
                        generate_video(
                            video_option=config.habitat_baselines.eval.video_option,
                            video_dir=config.habitat_baselines.video_dir,
                            # Since the final frame is the start frame of the next episode.
                            images=rgb_frames[i][:-1],
                            episode_id=f"{current_episodes_info[i].episode_id}_{ep_eval_count[k]}",
                            checkpoint_idx=checkpoint_index,
                            metrics=extract_scalars_from_info(disp_info),
                            fps=config.habitat_baselines.video_fps,
                            tb_writer=writer,
                            keys_to_include_in_name=config.habitat_baselines.eval_keys_to_include_in_name,
                        )

                        # Since the starting frame of the next episode is the final frame.
                        rgb_frames[i] = rgb_frames[i][-1:]

                    gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                    if gfx_str != "":
                        write_gfx_replay(
                            gfx_str,
                            config.habitat.task,
                            current_episodes_info[i].episode_id,
                        )

            not_done_masks = not_done_masks.to(device=device)
            (
                envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = pause_envs(
                envs_to_pause,
                envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        pbar.close()
        assert (
            len(ep_eval_count) >= number_of_eval_episodes
        ), f"Expected {number_of_eval_episodes} episodes, got {len(ep_eval_count)}."

        aggregated_stats = {}
        all_ks = set()
        for ep in stats_episodes.values():
            all_ks.update(ep.keys())
        for stat_key in all_ks:
            aggregated_stats[stat_key] = np.mean(
                [v[stat_key] for v in stats_episodes.values() if stat_key in v]
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        writer.add_scalar(
            "eval_reward/average_reward", aggregated_stats["reward"], step_id
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

    def _adapt(self,agent,
                mom_pre,
                decay_factor,
                min_mom):
    
        encoder = agent.actor_critic.visual_encoder
        mom_new = (mom_pre * decay_factor)
        min_momentum_constant = min_mom
        for m in encoder.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.train()
                m.momentum = mom_new + min_momentum_constant
        mom_pre = mom_new
        return agent, mom_pre, decay_factor, min_mom

    def _adapt_ablation(self,agent,
                mom_pre,
                decay_factor,
                min_mom):
    
        encoder = agent.actor_critic.visual_encoder
        mom_new = (mom_pre * decay_factor)
        min_momentum_constant = min_mom
        encoder_block = getattr(encoder.backbone, f"layer{self.ablation_block}")
        for m in encoder_block.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.train()
                m.momentum = mom_new + min_momentum_constant
        mom_pre = mom_new
        return agent, mom_pre, decay_factor, min_mom


# For 256 x 256 reconstructed images, they have to be resize to 224 x 224 for ViT models
def check_size_for_vit(observations, proper_im_size):
    for i in range(len(observations)):
        if observations[i]["rgb"].shape[1] != proper_im_size:
            observations[i]["rgb"] = cv2.resize(observations[i]["rgb"], 
                                                dsize=(proper_im_size, proper_im_size), 
                                                interpolation = cv2.INTER_LINEAR)
    return observations
