"""Video recorder for Lunar Lander policies using RGB array frames, cosmetic sci-fi rocket enhancements, telemetry overlays, and MoviePy."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw

try:
    from moviepy import ImageSequenceClip
except ImportError:
    try:
        from moviepy.editor import ImageSequenceClip
    except ImportError:
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from src.lunar_rl.agents.model_manager import ModelManager
from src.lunar_rl.agents.ppo_agent import PPOAgent
from src.lunar_rl.environment.factory import make_lunar_environment
from src.lunar_rl.environment.random_agent import RandomAgent
from src.lunar_rl.utils.config_loader import load_config
from src.lunar_rl.utils.logger import get_logger
from src.lunar_rl.utils.reproducibility import set_seed
from src.lunar_rl.visualization.cosmetic_renderer import (
    apply_rocket_theme,
    enhance_frame_visuals,
)

logger = get_logger(__name__)


def create_transition_card(
    title: str,
    subtitle: str,
    status_tag: str = "AUTONOMOUS FLIGHT CONTROLLER",
    width: int = 600,
    height: int = 400,
    duration_frames: int = 50,
    tag_color: Tuple[int, int, int] = (100, 180, 255),
) -> List[np.ndarray]:
    """Generates an aesthetic high-tech transition card sequence for multi-attempt videos."""
    img = Image.new("RGB", (width, height), color=(8, 14, 26))
    draw = ImageDraw.Draw(img)

    # Decorative tech borders
    draw.rectangle([(15, 15), (width - 15, height - 15)], outline=(30, 58, 110), width=2)
    draw.rectangle([(20, 20), (width - 20, height - 20)], outline=(15, 30, 60), width=1)

    # Status badge
    draw.text((width // 2 - 140, 95), status_tag, fill=tag_color)

    # Main Title
    draw.text((width // 2 - 140, 165), title, fill=(255, 255, 255))

    # Subtitle
    draw.text((width // 2 - 170, 215), subtitle, fill=(180, 205, 230))

    frame_arr = np.array(img)
    return [frame_arr.copy() for _ in range(duration_frames)]


def add_telemetry_overlay(
    frame: np.ndarray,
    episode_label: str,
    step: int,
    cumulative_reward: float,
    speed: float,
    outcome_banner: Optional[str] = None,
    banner_color: Tuple[int, int, int] = (220, 50, 50),
) -> np.ndarray:
    """Renders a telemetry header and outcome banner over environment frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Top dark HUD bar
    draw.rectangle([(0, 0), (w, 36)], fill=(10, 16, 28))

    # HUD text
    draw.text((12, 4), f"{episode_label}  |  Step: {step}", fill=(255, 255, 255))
    draw.text(
        (12, 19),
        f"Return: {cumulative_reward:+6.1f}  |  Descent Speed: {speed:4.2f}",
        fill=(120, 200, 255),
    )

    # Optional outcome banner at bottom if episode terminated
    if outcome_banner:
        draw.rectangle([(0, h - 45), (w, h)], fill=banner_color)
        draw.text((w // 2 - 160, h - 32), outcome_banner, fill=(255, 255, 255))

    return np.array(img)


class VideoRecorder:
    """Records gameplay videos for trained agents and random baselines."""

    def __init__(self, output_dir: Path = Path("results/videos")) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_manager = ModelManager()

    def record_frames_to_mp4(
        self,
        frames: List[np.ndarray],
        output_filename: str,
        fps: int = 50,
    ) -> Path:
        """Writes frame list to an MP4 video file using MoviePy."""
        if not output_filename.endswith(".mp4"):
            output_filename = f"{output_filename}.mp4"
        save_path = self.output_dir / output_filename

        logger.info(f"Encoding {len(frames)} frames into video: {save_path}...")
        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_videofile(
            str(save_path),
            codec="libx264",
            audio=False,
            logger=None,
        )
        logger.info(f"Video saved successfully to {save_path}")
        return save_path

    def _render_single_episode(
        self,
        agent: Any,
        seed: int,
        label: str,
        fps: int = 50,
        deterministic: bool = True,
    ) -> Tuple[List[np.ndarray], float, bool, bool]:
        """Runs and renders a single episode with sci-fi cosmetic rocket enhancements and HUD overlays."""
        env = make_lunar_environment(
            env_id="LunarLander-v3",
            render_mode="rgb_array",
            seed=seed,
        )

        frames: List[np.ndarray] = []
        obs, info = env.reset(seed=seed)

        # Apply custom sleek rocket colors to Box2D bodies
        apply_rocket_theme(env)

        first_raw = env.render()
        if first_raw is not None:
            first_styled = enhance_frame_visuals(env, first_raw, action=0, step=0)
            first_hud = add_telemetry_overlay(first_styled, label, 0, 0.0, 0.0)
            frames.append(first_hud)

        ep_reward = 0.0
        steps = 0
        done = False
        is_success = False
        is_crash = False

        try:
            while not done:
                pred_res = agent.predict(obs, deterministic=deterministic)
                action = pred_res[0] if isinstance(pred_res, tuple) else int(pred_res)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                steps += 1
                done = terminated or truncated

                speed = float(np.sqrt(obs[2] ** 2 + obs[3] ** 2))
                raw_frame = env.render()

                outcome_text = None
                banner_color = (12, 18, 32)
                if terminated:
                    if reward >= 100:
                        is_success = True
                        outcome_text = "PERFECT SAFE LANDING! (SUCCESS +100)"
                        banner_color = (25, 135, 84)
                    elif reward <= -100:
                        is_crash = True
                        outcome_text = "CRASH IMPACT DETECTED (TERMINAL PENALTY)"
                        banner_color = (220, 53, 69)

                if raw_frame is not None:
                    # Apply cosmetic rocket enhancements (canopy, exhaust plumes, runway lights)
                    styled_frame = enhance_frame_visuals(
                        env, raw_frame, action=action, step=steps
                    )
                    overlay_frame = add_telemetry_overlay(
                        styled_frame,
                        label,
                        steps,
                        ep_reward,
                        speed,
                        outcome_banner=outcome_text,
                        banner_color=banner_color,
                    )
                    frames.append(overlay_frame)

            # Freeze frames at end so viewer can absorb the outcome
            if frames:
                freeze_count = 50 if is_success else 35
                last = frames[-1]
                for _ in range(freeze_count):
                    frames.append(last)

        finally:
            env.close()

        return frames, ep_reward, is_success, is_crash

    def record_multi_attempt_journey(
        self,
        agent: Any,
        fail_seed_1: int = 26,
        fail_seed_2: int = 27,
        success_seed: int = 24,
        combined_filename: str = "continuous_journey_landing.mp4",
        fps: int = 50,
    ) -> Dict[str, Path]:
        """Captures 2 failed landing attempts followed by a perfect safe landing.

        Produces:
            1. fail_attempt_1.mp4: Crash attempt #1
            2. fail_attempt_2.mp4: Crash/gear collapse attempt #2
            3. perfect_safe_landing.mp4: Flawless touchdown attempt #3
            4. continuous_journey_landing.mp4: All 3 attempts stitched with transition title cards!
        """
        paths: Dict[str, Path] = {}

        # 1. Render Failure Attempt 1 (Crash on terrain)
        logger.info(f"Rendering Attempt 1: Failure / Crash (seed={fail_seed_1})...")
        fail1_frames, fail1_rew, _, _ = self._render_single_episode(
            agent=agent,
            seed=fail_seed_1,
            label="ATTEMPT 1: FAILED ATTEMPT (CRASH IMPACT)",
            fps=fps,
            deterministic=True,
        )
        if fail1_frames:
            paths["fail_1"] = self.record_frames_to_mp4(
                fail1_frames, "fail_attempt_1.mp4", fps=fps
            )

        # 2. Render Failure Attempt 2 (Off-target lateral drift / tip-over)
        logger.info(f"Rendering Attempt 2: Failure / Off-target (seed={fail_seed_2})...")
        fail2_frames, fail2_rew, _, _ = self._render_single_episode(
            agent=agent,
            seed=fail_seed_2,
            label="ATTEMPT 2: FAILED ATTEMPT (GEAR COLLAPSE)",
            fps=fps,
            deterministic=True,
        )
        if fail2_frames:
            paths["fail_2"] = self.record_frames_to_mp4(
                fail2_frames, "fail_attempt_2.mp4", fps=fps
            )

        # 3. Render Attempt 3: Perfect Safe Landing
        logger.info(f"Rendering Attempt 3: Perfect Safe Landing (seed={success_seed})...")
        succ_frames, succ_rew, is_succ, _ = self._render_single_episode(
            agent=agent,
            seed=success_seed,
            label="ATTEMPT 3: AUTONOMOUS GUIDANCE -> SAFE LANDING",
            fps=fps,
            deterministic=True,
        )
        if succ_frames:
            paths["success"] = self.record_frames_to_mp4(
                succ_frames, "perfect_safe_landing.mp4", fps=fps
            )

        # 4. Assemble the complete journey video: 2 fails + correct landing
        if fail1_frames and fail2_frames and succ_frames:
            logger.info("Assembling all-in-one cinematic journey: 2 fails + correct landing...")
            w, h = fail1_frames[0].shape[1], fail1_frames[0].shape[0]

            intro_card = create_transition_card(
                title="LUNAR LANDER RL LAB",
                subtitle="Episode 1: Flight Controller Failure Attempt",
                status_tag="MISSION START — ATTEMPT 1",
                width=w,
                height=h,
                duration_frames=50,
                tag_color=(239, 68, 68),
            )

            retry_card_1 = create_transition_card(
                title="ATTEMPT 1 TERMINATED (CRASH -100)",
                subtitle="Touchdown kinetic energy exceeded limits -> Retrying...",
                status_tag="MISSION ABORT — RE-CALIBRATING",
                width=w,
                height=h,
                duration_frames=55,
                tag_color=(245, 158, 11),
            )

            retry_card_2 = create_transition_card(
                title="ATTEMPT 2 TERMINATED (GEAR COLLAPSE)",
                subtitle="Activating Converged PPO Controller for Precision Landing...",
                status_tag="FINAL DESCENT LOCK — ATTEMPT 3",
                width=w,
                height=h,
                duration_frames=55,
                tag_color=(56, 189, 248),
            )

            success_card = create_transition_card(
                title="MISSION ACCOMPLISHED",
                subtitle=f"Safe Touchdown on Target Helipad Confirmed (Return: {succ_rew:+.1f})",
                status_tag="PERFECT SAFE LANDING",
                width=w,
                height=h,
                duration_frames=65,
                tag_color=(34, 197, 94),
            )

            combined_frames = (
                intro_card
                + fail1_frames
                + retry_card_1
                + fail2_frames
                + retry_card_2
                + succ_frames
                + success_card
            )

            paths["combined"] = self.record_frames_to_mp4(
                combined_frames, combined_filename, fps=fps
            )

        return paths

    def record_agent_gameplay(
        self,
        agent: Any,
        video_filename: str,
        episodes: int = 1,
        fps: int = 50,
        deterministic: bool = True,
        seed: int = 42,
    ) -> Path:
        """Runs policy in rgb_array render mode and writes video frames to MP4 file."""
        all_frames: List[np.ndarray] = []
        for ep in range(1, episodes + 1):
            cur_seed = seed + ep
            label = f"EPISODE {ep}/{episodes}"
            frames, _, _, _ = self._render_single_episode(
                agent=agent,
                seed=cur_seed,
                label=label,
                fps=fps,
                deterministic=deterministic,
            )
            all_frames.extend(frames)

        return self.record_frames_to_mp4(all_frames, video_filename, fps=fps)

    def record_experiment(
        self,
        experiment_name: str,
        episodes: int = 1,
        fps: int = 50,
        deterministic: bool = True,
        seed: int = 42,
    ) -> Path:
        """Convenience method to record a trained experiment model."""
        if experiment_name == "random":
            env = make_lunar_environment(seed=seed)
            agent = RandomAgent(env.action_space)
            env.close()
            return self.record_agent_gameplay(
                agent=agent,
                video_filename="random_agent.mp4",
                episodes=episodes,
                fps=fps,
                deterministic=False,
                seed=seed,
            )
        else:
            model = self.model_manager.load_model(experiment_name)
            cfg = load_config(experiment_name=experiment_name)
            agent = PPOAgent(config=cfg, model=model)
            return self.record_agent_gameplay(
                agent=agent,
                video_filename=f"{experiment_name}_gameplay.mp4",
                episodes=episodes,
                fps=fps,
                deterministic=deterministic,
                seed=seed,
            )
