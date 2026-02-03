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

SEQUENCES = ["00", "05", "07"]  # All sequences to process

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

# =====================================================
# TRAJECTORY ALIGNMENT (Sim3 - Umeyama Method)
# =====================================================
def align_trajectory_sim3(est, gt):
    """
    Align estimated trajectory to ground truth using Sim3 transformation
    (Scale + Rotation + Translation)
    This is the Umeyama algorithm for point set alignment
    """
    # Center both point clouds
    est_center = est.mean(axis=0)
    gt_center = gt.mean(axis=0)
    
    est_centered = est - est_center
    gt_centered = gt - gt_center
    
    # Compute scale
    est_scale = np.sqrt(np.mean(np.sum(est_centered**2, axis=1)))
    gt_scale = np.sqrt(np.mean(np.sum(gt_centered**2, axis=1)))
    scale = gt_scale / (est_scale + 1e-8)
    
    # Scale the estimated trajectory
    est_scaled = est_centered * scale
    
    # Compute rotation using SVD
    H = est_scaled.T @ gt_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure right-handed coordinate system (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Apply transformation: scale, rotate, then translate
    est_aligned = (R @ est_scaled.T).T + gt_center
    
    return est_aligned, scale, R

# =====================================================
# RPE CALCULATION
# =====================================================
def compute_rpe(traj, gt):
    """Compute Relative Pose Error (RPE) between trajectory and ground truth."""
    errors = []
    for i in range(1, len(traj)):
        dp_est = traj[i] - traj[i - 1]
        dp_gt = gt[i] - gt[i - 1]
        errors.append(np.linalg.norm(dp_est - dp_gt))
    return np.mean(errors)

# =====================================================
# PROCESS EACH SEQUENCE
# =====================================================
all_results = []

for SEQ in SEQUENCES:
    
    print(f"\n{'='*60}")
    print(f"Running ORB Stereo VO on KITTI sequence {SEQ}")
    print(f"{'='*60}")
    
    SEQ_PATH  = f"{BASE_PATH}/sequences/{SEQ}"
    LEFT_DIR  = f"{SEQ_PATH}/image_0"
    RIGHT_DIR = f"{SEQ_PATH}/image_1"
    CALIB     = f"{SEQ_PATH}/calib.txt"
    GT_PATH   = f"{POSES_ROOT}/{SEQ}.txt"

    K, BASELINE = load_kitti_calib(CALIB)
    GT_POSES = load_gt(GT_PATH)

    print(f"Total frames: {len(GT_POSES)}")
    print(f"Baseline: {BASELINE:.4f} m\n")

    # =====================================================
    # FEATURE SETUP
    # =====================================================
    orb = cv2.ORB_create(3000)
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # =====================================================
    # LOAD IMAGES
    # =====================================================
    left_imgs  = sorted(os.listdir(LEFT_DIR))
    right_imgs = sorted(os.listdir(RIGHT_DIR))
    assert len(left_imgs) == len(right_imgs)

    # =====================================================
    # INITIALIZE
    # =====================================================
    pose = np.eye(4)
    traj_est = [pose[:3, 3].copy()]

    prev_des = None
    prev_pts3D = None
    prev_kp = None

    times = []
    num_3d_list = []
    num_inliers_list = []

    # =====================================================
    # MAIN LOOP
    # =====================================================
    for i in range(len(left_imgs)):

        t0 = time.time()

        imgL = cv2.imread(os.path.join(LEFT_DIR, left_imgs[i]), 0)
        imgR = cv2.imread(os.path.join(RIGHT_DIR, right_imgs[i]), 0)

        kpL, desL = orb.detectAndCompute(imgL, None)
        kpR, desR = orb.detectAndCompute(imgR, None)

        if desL is None or desR is None or len(kpL) < 10 or len(kpR) < 10:
            traj_est.append(pose[:3,3].copy())
            times.append((time.time() - t0) * 1000)
            num_3d_list.append(0)
            num_inliers_list.append(0)
            continue

        matches_stereo = bf.knnMatch(desL, desR, k=2)

        good_stereo = []
        for m, n in matches_stereo:
            if m.distance < 0.7 * n.distance:
                good_stereo.append(m)

        if len(good_stereo) < 20:
            traj_est.append(pose[:3,3].copy())
            times.append((time.time() - t0) * 1000)
            num_3d_list.append(0)
            num_inliers_list.append(0)
            continue

        pts3D = []
        valid_desL = []
        valid_kpL = []

        for m in good_stereo:
            ptL = kpL[m.queryIdx].pt
            ptR = kpR[m.trainIdx].pt
            disparity = ptL[0] - ptR[0]

            if disparity > 1.0:
                Z = (K[0,0] * BASELINE) / disparity
                if 0.1 < Z < 100:
                    X = (ptL[0] - K[0,2]) * Z / K[0,0]
                    Y = (ptL[1] - K[1,2]) * Z / K[1,1]
                    pts3D.append([X, Y, Z])
                    valid_desL.append(desL[m.queryIdx])
                    valid_kpL.append(kpL[m.queryIdx])

        if len(pts3D) < 20:
            traj_est.append(pose[:3,3].copy())
            times.append((time.time() - t0) * 1000)
            num_3d_list.append(0)
            num_inliers_list.append(0)
            continue

        pts3D = np.array(pts3D, dtype=np.float32)
        valid_desL = np.array(valid_desL)
        num_3d_list.append(len(pts3D))

        num_inliers = 0
        if prev_pts3D is not None and prev_des is not None:
            matches_temporal = bf.knnMatch(prev_des, valid_desL, k=2)

            obj_pts = []
            img_pts = []

            for m, n in matches_temporal:
                if m.distance < 0.7 * n.distance:
                    obj_pts.append(prev_pts3D[m.queryIdx])
                    img_pts.append(valid_kpL[m.trainIdx].pt)

            if len(obj_pts) >= 8:
                obj_pts = np.array(obj_pts, dtype=np.float32)
                img_pts = np.array(img_pts, dtype=np.float32)

                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    obj_pts, img_pts, K, None,
                    iterationsCount=200,
                    reprojectionError=2.0,
                    confidence=0.999
                )

                if success and inliers is not None and len(inliers) > 6:
                    num_inliers = len(inliers)
                    R, _ = cv2.Rodrigues(rvec)
                    T_rel = np.eye(4)
                    T_rel[:3,:3] = R
                    T_rel[:3,3]  = tvec.squeeze()
                    pose = pose @ T_rel

        num_inliers_list.append(num_inliers)
        prev_des = valid_desL
        prev_pts3D = pts3D
        prev_kp = valid_kpL

        traj_est.append(pose[:3,3].copy())
        times.append((time.time() - t0) * 1000)

        if i % 200 == 0:
            print(f"Processed frame {i}/{len(left_imgs)}")

    print(f"Processed frame {len(left_imgs)}/{len(left_imgs)}")

    # =====================================================
    # EVALUATION
    # =====================================================
    traj_est = np.array(traj_est)
    
    # Ensure both trajectories have the same length
    min_len = min(len(traj_est), len(GT_POSES))
    traj_est = traj_est[:min_len]
    traj_gt  = GT_POSES[:min_len, :3, 3]

    traj_est_aligned, scale, R = align_trajectory_sim3(traj_est, traj_gt)

    ate = np.sqrt(np.mean(np.sum((traj_est_aligned - traj_gt)**2, axis=1)))
    rpe = compute_rpe(traj_est_aligned, traj_gt)

    avg_time = np.mean(times)
    avg_fps  = 1000.0 / avg_time
    avg_3d   = np.mean([x for x in num_3d_list if x > 0])
    avg_inliers = np.mean([x for x in num_inliers_list if x > 0])

    print("\n================ FINAL RESULTS ================")
    print(f"Sequence            : {SEQ}")
    print(f"Total frames        : {len(traj_est)}")
    print(f"ATE RMSE (m)        : {ate:.3f}")
    print(f"RPE (m)             : {rpe:.3f}")
    print(f"Scale factor        : {scale:.4f}")
    print(f"Avg runtime (ms)    : {avg_time:.2f}")
    print(f"Avg FPS             : {avg_fps:.2f}")
    print(f"Avg 3D points/frame : {avg_3d:.0f}")
    print(f"Avg inliers/frame   : {avg_inliers:.0f}")
    print("==============================================")

    # Store results
    all_results.append({
        'sequence': SEQ,
        'frames': len(traj_est),
        'ate': ate,
        'rpe': rpe,
        'scale': scale,
        'avg_time': avg_time,
        'avg_fps': avg_fps,
        'avg_3d': avg_3d,
        'avg_inliers': avg_inliers
    })

    # =====================================================
    # TRAJECTORY PLOT WITH BETTER VISUALIZATION
    # =====================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Main trajectory plot
    ax1.plot(traj_gt[:,0], traj_gt[:,2], 'b-', label="Ground Truth", linewidth=2, alpha=0.8)
    ax1.plot(traj_est_aligned[:,0], traj_est_aligned[:,2], 'r--', label="ORB Stereo VO", linewidth=2, alpha=0.8)
    ax1.scatter(traj_gt[0,0], traj_gt[0,2], c='green', s=100, marker='o', label='Start', zorder=5)
    ax1.scatter(traj_gt[-1,0], traj_gt[-1,2], c='red', s=100, marker='X', label='End', zorder=5)
    ax1.legend(fontsize=11, loc='best')
    ax1.set_xlabel("X [m]", fontsize=12)
    ax1.set_ylabel("Z [m]", fontsize=12)
    ax1.set_title(f"Trajectory Comparison - Sequence {SEQ}", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axis("equal")
    
    # Error plot over trajectory
    errors = np.sqrt(np.sum((traj_est_aligned - traj_gt)**2, axis=1))
    sc = ax2.scatter(traj_gt[:,0], traj_gt[:,2], c=errors, cmap='hot', s=20, alpha=0.7)
    ax2.set_xlabel("X [m]", fontsize=12)
    ax2.set_ylabel("Z [m]", fontsize=12)
    ax2.set_title(f"Position Error Heatmap - Sequence {SEQ}", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axis("equal")
    cbar = plt.colorbar(sc, ax=ax2)
    cbar.set_label('Error [m]', fontsize=11)
    
    # Add stats text box
    textstr = f'ATE: {ate:.2f}m\nRPE: {rpe:.3f}m\nScale: {scale:.4f}\nFPS: {avg_fps:.1f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(f"traj_{SEQ}_fixed.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nTrajectory plot saved as: traj_{SEQ}_fixed.png")
    print("Successfully completed")

# =====================================================
# SUMMARY TABLE
# =====================================================
print("\n" + "="*90)
print("SUMMARY OF ALL SEQUENCES")
print("="*90)
print(f"{'Seq':<6} {'Frames':<8} {'ATE (m)':<10} {'RPE (m)':<10} {'Scale':<8} {'Time (ms)':<12} {'FPS':<8} {'3D pts':<8} {'Inliers':<8}")
print("-" * 90)

for r in all_results:
    print(f"{r['sequence']:<6} {r['frames']:<8} {r['ate']:<10.3f} {r['rpe']:<10.3f} {r['scale']:<8.4f} {r['avg_time']:<12.2f} {r['avg_fps']:<8.2f} {r['avg_3d']:<8.0f} {r['avg_inliers']:<8.0f}")

print("="*90)
print("\nAll sequences completed successfully!")
print(f"Trajectory plots saved for sequences: {', '.join(SEQUENCES)}")
