import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

# =====================================================
# PATHS (CONFIRMED)
# =====================================================
BASE_PATH  = "/home/drdo/nikhitha/dataset1/dataset"
POSES_ROOT = "/home/drdo/nikhitha/poses1/poses"

SEQUENCES = ["00", "05", "07"]

# =====================================================
# LOAD CALIBRATION
# =====================================================
def load_kitti_calib(calib_file):
    with open(calib_file) as f:
        for line in f:
            if line.startswith("P0"):
                P0 = np.array(list(map(float, line.split()[1:]))).reshape(3,4)
            if line.startswith("P1"):
                P1 = np.array(list(map(float, line.split()[1:]))).reshape(3,4)

    K = P0[:, :3]
    baseline = abs(P1[0, 3] - P0[0, 3]) / K[0, 0]
    return K, baseline

# =====================================================
# LOAD GROUND TRUTH
# =====================================================
def load_gt(path):
    poses = []
    with open(path) as f:
        for line in f:
            T = np.eye(4)
            vals = list(map(float, line.split()))
            T[:3, :4] = np.array(vals).reshape(3,4)
            poses.append(T)
    return np.array(poses)
def align_trajectory_sim3(est, gt):
    # =====================================================
    # FIX: enforce equal trajectory length
    # =====================================================
    n = min(len(est), len(gt))
    est = est[:n]
    gt  = gt[:n]

    # =====================================================
    # Center trajectories
    # =====================================================
    est_center = est.mean(axis=0)
    gt_center  = gt.mean(axis=0)

    est_c = est - est_center
    gt_c  = gt - gt_center

    # =====================================================
    # Scale (Umeyama)
    # =====================================================
    scale = np.sqrt(np.mean(np.sum(gt_c**2, axis=1))) / \
            (np.sqrt(np.mean(np.sum(est_c**2, axis=1))) + 1e-8)

    est_c *= scale

    # =====================================================
    # Rotation
    # =====================================================
    H = est_c.T @ gt_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # =====================================================
    # Apply alignment
    # =====================================================
    est_aligned = (R @ est_c.T).T + gt_center

    return est_aligned, scale, R


# =====================================================
# RPE
# =====================================================
def compute_rpe(traj, gt):
    errors = []
    for i in range(1, len(traj)):
        errors.append(np.linalg.norm((traj[i] - traj[i-1]) -
                                     (gt[i]   - gt[i-1])))
    return np.mean(errors)

# =====================================================
# RoMA-style Stereo Matching
# =====================================================
def roma_feature_matching(imgL, imgR, K, baseline):
    detector = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2)

    kpL, desL = detector.detectAndCompute(imgL, None)
    kpR, desR = detector.detectAndCompute(imgR, None)

    if desL is None or desR is None or len(kpL) < 10:
        return None, None, None

    matches = bf.knnMatch(desL, desR, k=2)
    pts3D, des_valid, kp_valid = [], [], []

    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            xl, yl = kpL[m.queryIdx].pt
            xr, _  = kpR[m.trainIdx].pt
            disp = xl - xr

            if disp > 1.0:
                Z = K[0,0] * baseline / disp
                if 0.1 < Z < 100:
                    X = (xl - K[0,2]) * Z / K[0,0]
                    Y = (yl - K[1,2]) * Z / K[1,1]
                    pts3D.append([X, Y, Z])
                    des_valid.append(desL[m.queryIdx])
                    kp_valid.append(kpL[m.queryIdx])

    if len(pts3D) < 20:
        return None, None, None

    return np.array(pts3D, np.float32), np.array(des_valid), kp_valid

# =====================================================
# MAIN
# =====================================================
all_results = []

for SEQ in SEQUENCES:

    print(f"\n{'='*60}")
    print(f"Running RoMA Stereo VO on KITTI sequence {SEQ}")
    print(f"{'='*60}")

    SEQ_PATH  = f"{BASE_PATH}/sequences/{SEQ}"
    LEFT_DIR  = f"{SEQ_PATH}/image_0"
    RIGHT_DIR = f"{SEQ_PATH}/image_1"

    K, BASELINE = load_kitti_calib(f"{SEQ_PATH}/calib.txt")
    GT_POSES = load_gt(f"{POSES_ROOT}/{SEQ}.txt")

    pose = np.eye(4)
    traj_est = [pose[:3,3].copy()]

    prev_pts3D = prev_des = None
    times, num_3d_list, num_inliers_list = [], [], []

    files = sorted(os.listdir(LEFT_DIR))

    for i in range(len(files)):
        t0 = time.time()

        imgL = cv2.imread(os.path.join(LEFT_DIR, files[i]), 0)
        imgR = cv2.imread(os.path.join(RIGHT_DIR, files[i]), 0)

        pts3D, desL, kpL = roma_feature_matching(imgL, imgR, K, BASELINE)

        if pts3D is None:
            traj_est.append(pose[:3,3].copy())
            times.append((time.time() - t0) * 1000)
            num_3d_list.append(0)
            num_inliers_list.append(0)
            continue

        num_3d_list.append(len(pts3D))

        if prev_pts3D is not None:
            bf = cv2.BFMatcher(cv2.NORM_L2)
            matches = bf.knnMatch(prev_des, desL, k=2)

            obj, img = [], []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    obj.append(prev_pts3D[m.queryIdx])
                    img.append(kpL[m.trainIdx].pt)

            if len(obj) >= 8:
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    np.array(obj, np.float32),
                    np.array(img, np.float32),
                    K, None
                )

                if success and inliers is not None and len(inliers) > 6:
                    R, _ = cv2.Rodrigues(rvec)
                    T = np.eye(4)
                    T[:3,:3] = R
                    T[:3,3]  = tvec.squeeze()
                    pose = pose @ T
                    num_inliers_list.append(len(inliers))
                else:
                    num_inliers_list.append(0)
            else:
                num_inliers_list.append(0)
        else:
            num_inliers_list.append(0)

        prev_pts3D, prev_des = pts3D, desL
        traj_est.append(pose[:3,3].copy())
        times.append((time.time() - t0) * 1000)

        if i % 200 == 0:
            print(f"Processed frame {i}/{len(files)}")

    traj_est = np.array(traj_est)
    traj_gt  = GT_POSES[:len(traj_est), :3, 3]

    traj_aligned, scale, _ = align_trajectory_sim3(traj_est, traj_gt)

    ate = np.sqrt(np.mean(np.sum((traj_aligned - traj_gt)**2, axis=1)))
    rpe = compute_rpe(traj_aligned, traj_gt)

    avg_time = np.mean(times)
    avg_fps  = 1000.0 / avg_time

    print("\n================ FINAL RESULTS ================")
    print(f"Sequence            : {SEQ}")
    print(f"ATE RMSE (m)        : {ate:.3f}")
    print(f"RPE (m)             : {rpe:.3f}")
    print(f"Scale factor        : {scale:.4f}")
    print(f"Avg FPS             : {avg_fps:.2f}")
    print(f"Avg 3D points/frame : {np.mean([x for x in num_3d_list if x > 0]):.0f}")
    print(f"Avg inliers/frame   : {np.mean([x for x in num_inliers_list if x > 0]):.0f}")
    print("==============================================")

    # =====================================================
    # TRAJECTORY PLOT ONLY
    # =====================================================
    plt.figure(figsize=(10,7))
    plt.plot(traj_gt[:,0], traj_gt[:,2], 'b-', label="Ground Truth", linewidth=2)
    plt.plot(traj_aligned[:,0], traj_aligned[:,2], 'r--', label="RoMA Stereo VO", linewidth=2)
    plt.scatter(traj_gt[0,0], traj_gt[0,2], c='green', s=80, label='Start')
    plt.scatter(traj_gt[-1,0], traj_gt[-1,2], c='red', s=80, label='End')
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlabel("X [m]")
    plt.ylabel("Z [m]")
    plt.title(f"Trajectory Comparison - Sequence {SEQ}")
    plt.savefig(f"traj_{SEQ}_fixed.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Trajectory plot saved as: traj_{SEQ}_fixed.png")

print("\nAll sequences completed successfully!")
