import argparse
import os
import pprint

from .util import get_str_formatted_time, ensure_dir


def get_corruptions_parser():
    parser = argparse.ArgumentParser(
        description="corruptions", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-hni",
        "--habitat_rgb_noise_intensity",
        default=0.1,
        type=float,
        required=False,
        help="Intensity of RGB noise introduced by habitat (in the 2021 challenge it was GaussianNoiseModel with "
             "intensity 0.1). This allows the noise to be disabled by setting the intensity to 0.0, or reinforcing "
             "its magnitude by making the intensity higher.",
    )
    parser.add_argument(
        "-dnm",
        "--depth_noise_multiplier",
        default=1.0,
        type=float,
        required=False,
        help="Depth sensor noise multiplier. This corruptions assumes that the noise model has a noise_multiplier "
             "kwarg, as RedwoodDepthNoiseModel does.",
    )

    parser.add_argument(
        "-hfov",
        "--habitat_rgb_hfov",
        default=70,
        type=int,
        required=False,
        help="Habitat RGB sensor horizontal field of view (in the 2021 challenge, the default was 70)",
    )

    # Defocus_Blur Lighting Speckle_Noise Spatter Motion_Blur
    parser.add_argument(
        "-vc",
        "--visual_corruption",
        default=None,
        type=str,
        required=False,
        help="Visual corruption to be applied to egocentric RGB observation",
    )

    parser.add_argument(
        "-vs",
        "--visual_severity",
        default=0,
        type=int,
        required=False,
        help="Severity of visual corruption to be applied",
    )

    # parser.add_argument(
    #     "-dcr",
    #     "--dyn_corr_mode",
    #     dest="dyn_corr_mode",
    #     required=False,
    #     action="store_true",
    #     help="Whether to apply dynamics corruptions",
    # )
    # parser.set_defaults(dyn_corr_mode=False)
    #
    # parser.add_argument(
    #     "-mf",
    #     "--motor_failure",
    #     dest="motor_failure",
    #     required=False,
    #     action="store_true",
    #     help="Whether to apply motor failure as the dynamics corruption",
    # )
    # parser.set_defaults(motor_failure=False)
    #
    # parser.add_argument(
    #     "-ctr",
    #     "--const_translate",
    #     dest="const_translate",
    #     required=False,
    #     action="store_true",
    #     help="Whether to apply constant translation bias as the dynamics corruption",
    # )
    # parser.set_defaults(const_translate=False)
    #
    # parser.add_argument(
    #     "-crt",
    #     "--const_rotate",
    #     dest="const_rotate",
    #     required=False,
    #     action="store_true",
    #     help="Whether to apply constant rotation bias as the dynamics corruption",
    # )
    # parser.set_defaults(const_rotate=False)
    #
    # parser.add_argument(
    #     "-str",
    #     "--stoch_translate",
    #     dest="stoch_translate",
    #     required=False,
    #     action="store_true",
    #     help="Whether to apply stochastic translation bias as the dynamics corruption",
    # )
    # parser.set_defaults(stoch_translate=False)
    #
    # parser.add_argument(
    #     "-srt",
    #     "--stoch_rotate",
    #     dest="stoch_rotate",
    #     required=False,
    #     action="store_true",
    #     help="Whether to apply stochastic rotation bias as the dynamics corruption",
    # )
    # parser.set_defaults(stoch_rotate=False)
    #
    # parser.add_argument(
    #     "-dr",
    #     "--drift",
    #     dest="drift",
    #     required=False,
    #     action="store_true",
    #     help="Whether to apply drift in translation as the dynamics corruption",
    # )
    # parser.set_defaults(drift=False)
    #
    # parser.add_argument(
    #     "-dr_deg",
    #     "--drift_degrees",
    #     default=1.15,
    #     type=float,
    #     required=False,
    #     help="Drift angle for the motion-drift dynamics corruption",
    # )

    parser.add_argument(
        "-irc",
        "--random_crop",
        dest="random_crop",
        required=False,
        action="store_true",
        help="Specify if random crop is to be applied to the egocentric observations",
    )
    parser.add_argument(
        "-cw",
        "--crop_width",
        type=int,
        required=False,
        help="Specify if random crop width is to be applied to the egocentric observations",
    )
    parser.add_argument(
        "-ch",
        "--crop_height",
        type=int,
        required=False,
        help="Specify if random crop height is to be applied to the egocentric observations",
    )
    parser.set_defaults(random_crop=False)

    parser.add_argument(
        "-icj",
        "--color_jitter",
        dest="color_jitter",
        required=False,
        action="store_true",
        help="Specify if random crop is to be applied to the egocentric observations",
    )
    parser.set_defaults(color_jitter=False)

    parser.add_argument(
        "-irs",
        "--random_shift",
        dest="random_shift",
        required=False,
        action="store_true",
        help="Specify if random shift is to be applied to the egocentric observations",
    )
    parser.set_defaults(random_shift=False)

    # parser.add_argument(
    #     "-irs",
    #     "--rotate",
    #     dest="rotate",
    #     required=False,
    #     action="store_true",
    #     help="Specify if random rotations of the image should be applied",
    # )
    # parser.set_defaults(rotate=False)

    # LoCoBot, LoCoBot-Lite
    parser.add_argument(
        "-pn_robot",
        "--pyrobot_robot_spec",
        default="LoCoBot",
        type=str,
        required=False,
        help="Which robot specification to use for PyRobot (LoCoBot, LoCoBot-Lite)",
    )

    # ILQR, Proportional, ILQR
    parser.add_argument(
        "-pn_controller",
        "--pyrobot_controller_spec",
        default="Proportional",
        type=str,
        required=False,
        help="Which PyRobot controller specification to use (ILQR, Proportional, ILQR)",
    )

    parser.add_argument(
        "-pn_multiplier",
        "--pyrobot_noise_multiplier",
        default=0.5,
        type=float,
        required=False,
        help="PyRobot noise magnitude multiplier",
    )

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "-ne",
        "--num_episodes",
        default=-1,
        type=int,
        required=False,
        help="Number of episodes to run the benchmark for, or None.",
    )
    parser.add_argument(
        "-config",
        "--challenge_config_file",
        default=None,
        type=str,
        required=False,
        help="Habitat config that, if specified, overwrites the environmental variable CHALLENGE_CONFIG_FILE",
    )
    parser.add_argument(
        "-an",
        "--agent_name",
        default=None,
        type=str,
        required=False,
        help="Agent name, used for loging",
    )
    parser.add_argument(
        "-nes",
        "--num_episode_sample",
        default=-1,
        type=int,
        required=False,
        help="Number of episodes to sample from all episodes. -1 for no sampling",
    )
    parser.add_argument(
        "-ds",
        "--dataset_split",
        default=None,
        type=str,
        required=False,
        help="Which dataset split to use (train, val, val_mini)",
    )
    parser.add_argument(
        "-lf",
        "--log_folder",
        default="logs",
        type=str,
        required=False,
        help="Where to save evaluation logs",
    )
    parser.add_argument(
        "-vli",
        "--video_log_interval",
        default=9_999_999_999,
        type=int,
        required=False,
        help="How often to log per-episode videos of agent solving the task",
    )
    return parser


def apply_corruptions_to_config(args, task_config):
    task_config.defrost()
    task_config.RANDOM_SEED = args.seed
    task_config.SIMULATOR.RGB_SENSOR.HFOV = args.habitat_rgb_hfov
    task_config.SIMULATOR.RGB_SENSOR.NOISE_MODEL_KWARGS.intensity_constant = args.habitat_rgb_noise_intensity
    task_config.SIMULATOR.DEPTH_SENSOR.NOISE_MODEL_KWARGS.noise_multiplier = args.depth_noise_multiplier
    task_config.SIMULATOR.NOISE_MODEL.ROBOT = args.pyrobot_robot_spec
    task_config.SIMULATOR.NOISE_MODEL.CONTROLLER = args.pyrobot_controller_spec
    task_config.SIMULATOR.NOISE_MODEL.NOISE_MULTIPLIER = args.pyrobot_noise_multiplier
    task_config.DATASET.SPLIT = args.dataset_split
    task_config.ENVIRONMENT.ITERATOR_OPTIONS.NUM_EPISODE_SAMPLE = args.num_episode_sample
    task_config.EASTER_EGG = 72
    task_config.freeze()


def get_runid_and_logfolder(args, task_config):
    active_corruptions_1 = f"pc={args.pyrobot_controller_spec}" \
                           f"_pr={args.pyrobot_robot_spec}" \
                           f"_pnm={args.pyrobot_noise_multiplier}" \
                           f"__habitatrgbnoise={args.habitat_rgb_noise_intensity}" \
                           f"_depthnoise={args.depth_noise_multiplier}"

    active_corruptions_2 = "_"
    if args.visual_corruption and args.visual_severity != 0:
        active_corruptions_2 += f"_{args.visual_corruption}={args.visual_severity}"
    if args.color_jitter:
        active_corruptions_2 += "+colorjitter"
    if args.random_crop:
        active_corruptions_2 += f"+radnomcrop={args.crop_width}x{args.crop_height}"
    if args.random_shift:
        active_corruptions_2 += f"+randomshift"
    if args.habitat_rgb_hfov != 70:
        active_corruptions_2 += f"+hfov={args.habitat_rgb_hfov}"

    runid = f"{get_str_formatted_time()}" \
            f"__{args.agent_name}" \
            f"__{task_config.DATASET.SPLIT}" \
            f"_{active_corruptions_2}" \
            f"__{active_corruptions_1}"

    logfolder = os.path.join(args.log_folder, args.agent_name)
    logfolder = os.path.join(logfolder, active_corruptions_2)
    logfolder = os.path.join(logfolder, runid)
    # logfolder = os.path.join(task_config.LOG_FOLDER, runid)
    ensure_dir(logfolder)

    with open(os.path.join(logfolder, "args.txt"), "w") as f:
        pprint.pprint(args.__dict__, stream=f)
    with open(os.path.join(logfolder, "config.txt"), "w") as f:
        f.write(f"{task_config}")
    with open(os.path.join(logfolder, "args_and_config.txt"), "w") as f:
        pprint.pprint(args.__dict__, stream=f)
        f.write(";;;\n")
        f.write(f"{task_config}")

    print(f"runid:\n{runid}")
    print(f"logfolder:\n{os.path.abspath(logfolder)}")
    return runid, logfolder
