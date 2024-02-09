import os
import subprocess

directory = "/home/makramchahine/repos/gym-pybullet-drones/gym_pybullet_drones/examples/cl_realgs_d6_nonorm_ss2_200_9hzf_bm_px_td_nlsp_gn_nt_srf_150sf_irreg2_64_hyp_cfc_mleno_9hz_05sf_100act_doub/val"
# directory = "/home/makramchahine/repos/gym-pybullet-drones/gym_pybullet_drones/examples/cl_realgs_d6_nonorm_ss2_600_1_10hzf_bm_px_td_nlsp_gn_nt_srf_300sf_irreg2_64_hyp_cfc_mleno_9hz_05sf_100act_doub/val"
# directory = "/home/makramchahine/repos/gym-pybullet-drones/gym_pybullet_drones/examples/cl_realgs_d6_nonorm_ss2_600_9hzf_bm_px_td_nlsp_gn_nt_srf_300sf_irreg2_64_hyp_cfc_mleno_9hz_05sf_100act_doub/val"
# directory = "/home/makramchahine/repos/gym-pybullet-drones/gym_pybullet_drones/examples/cl_realgs_d6_nonorm_ss2_600_3hzf_bm_px_td_nlsp_gn_nt_srf_300sf_irreg2_64_hyp_lstm_mleno_3hz_05sf_100act/val"
# directory = "/home/makramchahine/repos/gym-pybullet-drones/gym_pybullet_drones/examples/cl_realgs_d6_nonorm_ss2_600_1_10hzf_bm_px_td_nlsp_gn_nt_pybullet_srf_300sf_irreg2_64_hyp_cfc_mleno_3hz_05sf_100act_doub/val"

video_filename = "video.mp4"
half_video_filename = "half_video.mp4"
combined_half_video_filename = "combined_half_video.mp4"

# 36 seconds for 2160

def create_half_videos():
    for run_i in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, run_i)):
            print(run_i)
            if video_filename in os.listdir(os.path.join(directory, run_i)) and half_video_filename not in os.listdir(os.path.join(directory, run_i)):
            # if video_filename in os.listdir(os.path.join(directory, run_i)):
                # create new video that is just the first half of the video
                # os.system("rm " + os.path.join(directory, run_i, half_video_filename))
                os.system("ffmpeg -i " + os.path.join(directory, run_i, video_filename) + " -ss 00:00:00 -t 00:00:36 -async 1 " + os.path.join(directory, run_i, half_video_filename))

def combine_videos(eval_dir, video_filename="video.mp4", combined_video_filename="combined_video.mp4"):
    video_paths = [os.path.join(eval_dir, f"{absolute_path}/{video_filename}") for absolute_path in sorted(os.listdir(eval_dir)) if os.path.isdir(os.path.join(eval_dir, absolute_path)) and f"{video_filename}" in os.listdir(os.path.join(eval_dir, absolute_path))]
    # concatenate all videos in video_paths
    with open("input.txt", "w") as f:
        for video_path in video_paths:
            f.write(f"file {video_path}\n")

    subprocess.run(["ffmpeg", "-f", "concat", "-safe", "0", "-i", "input.txt", "-c", "copy", f"{eval_dir}/{combined_video_filename}"])


create_half_videos()
combine_videos(directory, half_video_filename, combined_half_video_filename)