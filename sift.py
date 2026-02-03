
import os
import cv2
import numpy as np
import time
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# =====================================================
# CONFIG
# =====================================================
SEQUENCES = ["00", "05", "07"]
BASE = "/home/drdo/nikhitha/dataset1/dataset"
POSES = "/home/drdo/nikhitha/poses1/poses"

# =====================================================
# LOAD CALIBRATION
# =====================================================
def load_calib(f):
    with open(f) as fp:
        P0 = np.array(fp.readline().split()[1:], float).reshape(3,4)
        P1 = np.array(fp.readline().split()[1:], float).reshape(3,4)
    K = P0[:, :3]
    baseline = abs(P1[0,3] / P0[0,0])
    return K, baseline

# =====================================================
# LOAD GROUND TRUTH
# =====================================================
def load_gt(path):
    poses = []
    for l in open(path):
        T = np.eye(4)
        T[:3,:] = np.array(l.split(), float).reshape(3,4)
        poses.append(T)
    return poses

# =====================================================
# UMEYAMA ALIGNMENT (Sim(3)) – for ATE only
# =====================================================
def umeyama_alignment(X, Y):
    muX = X.mean(0)
    muY = Y.mean(0)
    Xc = X - muX
    Yc = Y - muY

    S = (Xc.T @ Yc) / X.shape[0]
    U, D, Vt = np.linalg.svd(S)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T

    varX = np.mean(np.sum(Xc**2, axis=1))
    scale = np.trace(np.diag(D)) / varX
    t = muY - scale * R @ muX

    return (scale * (R @ X.T).T + t)

# =====================================================
# RELATIVE POSE ERROR (RPE)
# =====================================================
def compute_rpe(gt_poses, est_poses):
    trans_err = []
    rot_err = []

    for i in range(1, len(gt_poses)):
        dT_gt  = np.linalg.inv(gt_poses[i-1]) @ gt_poses[i]
        dT_est = np.linalg.inv(est_poses[i-1]) @ est_poses[i]

        dT_err = np.linalg.inv(dT_gt) @ dT_est

        # Translation error
        trans_err.append(np.linalg.norm(dT_err[:3, 3]))

        # Rotation error (angle)
        R = dT_err[:3, :3]
        angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        rot_err.append(np.degrees(angle))

    return (
        np.sqrt(np.mean(np.square(trans_err))),
        np.sqrt(np.mean(np.square(rot_err)))
    )

# =====================================================
# TRIANGULATION
# =====================================================
def triangulate(kpl, kpr, matches, fx, fy, cx, cy, baseline):
    pts = {}
    for m in matches:
        uL, vL = kpl[m.queryIdx].pt
        uR, vR = kpr[m.trainIdx].pt

        if abs(vL - vR) > 1.0:
            continue

        disp = uL - uR
        if disp <= 1.0:
            continue

        Z = fx * baseline / disp
        if not (0.5 < Z < 80):
            continue

        X = (uL - cx) * Z / fx
        Y = (vL - cy) * Z / fy
        pts[m.queryIdx] = np.array([X, Y, Z], np.float32)

    return pts

# =====================================================
# PROCESS SINGLE SEQUENCE
# =====================================================
def process_sequence(SEQ):
    print(f"\n{'='*60}")
    print(f"PROCESSING SEQUENCE {SEQ}")
    print(f"{'='*60}")

    L_DIR = f"{BASE}/sequences/{SEQ}/image_0"
    R_DIR = f"{BASE}/sequences/{SEQ}/image_1"
    CALIB = f"{BASE}/sequences/{SEQ}/calib.txt"
    GT    = f"{POSES}/{SEQ}.txt"
    OUTPUT_PLOT = f"traj_sift_stereo_{SEQ}.png"

    K, baseline = load_calib(CALIB)
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    print(f"Baseline: {baseline:.4f} m")

    gt_poses = load_gt(GT)

    sift = cv2.SIFT_create(4000)
    bf = cv2.BFMatcher(cv2.NORM_L2)

    imgs = sorted(os.listdir(L_DIR))
    N = len(imgs)

    Twc = np.eye(4)
    poses_est, poses_gt = [], []
    times = []

    prev_desc, prev_3d = None, None

    print(f"Running CPU SIFT Stereo VO on KITTI {SEQ}")
    print(f"Total frames: {N}\n")

    for i, fname in enumerate(imgs):
        if i % 200 == 0:
            print(f"Processed frame {i}/{N}")

        t0 = time.time()

        imgL = cv2.imread(os.path.join(L_DIR, fname), 0)
        imgR = cv2.imread(os.path.join(R_DIR, fname), 0)

        kpL, dL = sift.detectAndCompute(imgL, None)
        kpR, dR = sift.detectAndCompute(imgR, None)

        if dL is not None and dR is not None:
            matches = bf.knnMatch(dL, dR, k=2)
            good_lr = [m for m,n in matches if m.distance < 0.6*n.distance]
            cur_3d = triangulate(kpL, kpR, good_lr, fx, fy, cx, cy, baseline)

            if prev_desc is not None and len(cur_3d) > 80:
                matches_prev = bf.knnMatch(prev_desc, dL, k=2)
                obj, img_pts = [], []

                for m,n in matches_prev:
                    if m.distance < 0.6*n.distance and m.queryIdx in prev_3d:
                        obj.append(prev_3d[m.queryIdx])
                        img_pts.append(kpL[m.trainIdx].pt)

                if len(obj) > 40:
                    obj = np.array(obj, np.float32)
                    img_pts = np.array(img_pts, np.float32)

                    ok, rvec, tvec, inl = cv2.solvePnPRansac(
                        obj, img_pts, K, None,
                        reprojectionError=3.0,
                        iterationsCount=200,
                        confidence=0.999
                    )

                    if ok and inl is not None and len(inl) > 30:
                        R,_ = cv2.Rodrigues(rvec)
                        T = np.eye(4)
                        T[:3,:3] = R
                        T[:3,3]  = tvec.flatten()
                        Twc = Twc @ np.linalg.inv(T)

            prev_desc, prev_3d = dL, cur_3d

        poses_est.append(Twc.copy())
        poses_gt.append(gt_poses[i])
        times.append((time.time() - t0) * 1000)

    # =====================================================
    # METRICS
    # =====================================================
    traj_est = np.array([T[:3,3] for T in poses_est])
    traj_gt  = np.array([T[:3,3] for T in poses_gt])

    traj_est -= traj_est[0]
    traj_gt  -= traj_gt[0]

    traj_est_aligned = umeyama_alignment(traj_est, traj_gt)

    ate = np.sqrt(np.mean(np.sum((traj_est_aligned - traj_gt)**2, axis=1)))
    rpe_t, rpe_r = compute_rpe(poses_gt, poses_est)

    len_gt  = np.sum(np.linalg.norm(np.diff(traj_gt, axis=0), axis=1))
    len_est = np.sum(np.linalg.norm(np.diff(traj_est_aligned, axis=0), axis=1))

    print("\n========== FINAL METRICS ==========")
    print(f"Sequence          : {SEQ}")
    print(f"ATE RMSE (m)      : {ate:.3f}")
    print(f"RPE trans (m)     : {rpe_t:.3f}")
    print(f"RPE rot (deg)     : {rpe_r:.3f}")
    print(f"GT length (m)     : {len_gt:.1f}")
    print(f"Est length (m)    : {len_est:.1f}")
    print(f"Avg runtime (ms)  : {np.mean(times):.2f}")
    print(f"Avg FPS           : {1000/np.mean(times):.2f}")
    print("==================================")

    # =====================================================
    # PLOT
    # =====================================================
    plt.figure(figsize=(10,6))
    plt.plot(traj_gt[:,0], traj_gt[:,2], lw=2, label="Ground Truth")
    plt.plot(traj_est_aligned[:,0], traj_est_aligned[:,2],
             lw=2, color="red", label="CPU SIFT Stereo VO")
    plt.axis("equal")
    plt.xlabel("X [m]")
    plt.ylabel("Z [m]")
    plt.title(f"KITTI {SEQ} – Stereo SIFT Visual Odometry")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)
    plt.close()

    print(f"Trajectory plot saved to {OUTPUT_PLOT}")

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    print("\nCPU SIFT STEREO VISUAL ODOMETRY – MULTI SEQUENCE\n")
    for seq in SEQUENCES:
        process_sequence(seq)
