import os
import time
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import kornia.feature as KF

# =====================================================
# CONFIG
# =====================================================
SEQS = ["00", "05", "07"]

BASE  = "/home/drdo/nikhitha/dataset1/dataset"
POSES = "/home/drdo/nikhitha/poses1/poses"

# GPU Configuration - FORCE CUDA CHECK
print("="*60)
print("CHECKING GPU AVAILABILITY")
print("="*60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    DEVICE = torch.device("cuda:0")
    USE_FP16 = True
    print("\n✓ Using GPU with FP16 acceleration")
else:
    print("\n✗ WARNING: CUDA not available! Running on CPU (will be slow)")
    print("Please check:")
    print("  1. NVIDIA GPU is installed")
    print("  2. CUDA toolkit is installed")
    print("  3. PyTorch with CUDA support: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    DEVICE = torch.device("cpu")
    USE_FP16 = False

print("="*60 + "\n")

MIN_PNP_INLIERS = 30
PIXEL_THR = 2.0

# =====================================================
# UTILS
# =====================================================
def load_calibration(calib_file):
    with open(calib_file) as f:
        for l in f:
            if l.startswith("P0:"):
                P0 = np.array(l.split()[1:], float).reshape(3, 4)
            if l.startswith("P1:"):
                P1 = np.array(l.split()[1:], float).reshape(3, 4)

    K = P0[:, :3]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    baseline = abs(P1[0, 3] - P0[0, 3]) / fx
    return K, fx, fy, cx, cy, baseline


def img_tensor(img, device, use_fp16=False):
    """Optimized image to tensor conversion with pinned memory for fast GPU transfer"""
    tensor = torch.from_numpy(img.copy()).float()
    tensor = tensor.unsqueeze(0).unsqueeze(0) / 255.0
    
    if device.type == 'cuda':
        # Use non_blocking for faster GPU transfer
        tensor = tensor.to(device, non_blocking=True)
    else:
        tensor = tensor.to(device)
    
    if use_fp16:
        tensor = tensor.half()
    
    return tensor


def path_length(x):
    return np.sum(np.linalg.norm(np.diff(x, axis=0), axis=1))


def umeyama_alignment(X, Y):
    """Umeyama alignment (Sim(3)) - aligns X to Y"""
    muX = X.mean(0)
    muY = Y.mean(0)
    
    X0 = X - muX
    Y0 = Y - muY
    
    ssX = (X0**2).sum()
    ssY = (Y0**2).sum()
    
    C = Y0.T @ X0 / X.shape[0]
    U, D, Vt = np.linalg.svd(C)
    S = np.eye(3)
    
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    
    R = U @ S @ Vt
    scale = np.trace(np.diag(D) @ S) / (ssX / X.shape[0])
    t = muY - scale * (R @ muX)
    X_aligned = scale * (R @ X.T).T + t
    
    return X_aligned


def compute_ate(est, gt):
    """Absolute Trajectory Error with Sim(3) alignment"""
    est_aligned = umeyama_alignment(est, gt)
    ate = np.sqrt(np.mean(np.sum((est_aligned - gt)**2, axis=1)))
    return ate, est_aligned


def compute_rpe(est_poses, gt_poses):
    """Relative Pose Error - translation and rotation"""
    trans_errors = []
    rot_errors = []
    
    for i in range(1, len(est_poses)):
        dT_gt = np.linalg.inv(gt_poses[i-1]) @ gt_poses[i]
        dT_est = np.linalg.inv(est_poses[i-1]) @ est_poses[i]
        dT_err = np.linalg.inv(dT_gt) @ dT_est
        
        trans_errors.append(np.linalg.norm(dT_err[:3, 3]))
        
        R_err = dT_err[:3, :3]
        trace = np.clip((np.trace(R_err) - 1) / 2, -1, 1)
        angle = np.arccos(trace)
        rot_errors.append(np.degrees(angle))
    
    rpe_trans = np.sqrt(np.mean(np.square(trans_errors)))
    rpe_rot = np.sqrt(np.mean(np.square(rot_errors)))
    
    return rpe_trans, rpe_rot


# =====================================================
# VECTORIZED TRIANGULATION (FASTER)
# =====================================================
def triangulate_vectorized(kpL, kpR, fx, fy, cx, cy, baseline):
    """Fast vectorized triangulation"""
    kpL = np.asarray(kpL)
    kpR = np.asarray(kpR)
    
    # Compute disparities
    disp = kpL[:, 0] - kpR[:, 0]
    
    # Filter valid disparities
    valid_mask = disp > 1.0
    
    if not np.any(valid_mask):
        return np.array([]), np.array([])
    
    kpL_valid = kpL[valid_mask]
    kpR_valid = kpR[valid_mask]
    disp_valid = disp[valid_mask]
    
    # Compute depths
    z = fx * baseline / disp_valid
    
    # Filter by depth range
    depth_mask = (z > 0.5) & (z < 80)
    
    if not np.any(depth_mask):
        return np.array([]), np.array([])
    
    kpL_valid = kpL_valid[depth_mask]
    z = z[depth_mask]
    
    # Compute 3D points
    x = (kpL_valid[:, 0] - cx) * z / fx
    y = (kpL_valid[:, 1] - cy) * z / fy
    
    pts3d = np.stack([x, y, z], axis=1).astype(np.float32)
    kpL_valid = kpL_valid.astype(np.float32)
    
    return pts3d, kpL_valid


# =====================================================
# LoFTR - GPU Optimized
# =====================================================
print("Loading LoFTR model...")
loftr = KF.LoFTR(pretrained=None).eval()

# Load checkpoint
ckpt = torch.load(
    "/home/drdo/nikhitha/weight1/weights/outdoor_ds.ckpt",
    map_location='cpu'  # Load to CPU first
)
loftr.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)

# Move to GPU and optimize
loftr = loftr.to(DEVICE)

if torch.cuda.is_available():
    # Enable cudnn optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Convert to half precision
    if USE_FP16:
        loftr = loftr.half()
        print("Model converted to FP16 for faster inference")
    
    # Warm up GPU
    print("Warming up GPU...")
    dummy_img = torch.zeros(1, 1, 480, 640).to(DEVICE)
    if USE_FP16:
        dummy_img = dummy_img.half()
    with torch.no_grad():
        _ = loftr({"image0": dummy_img, "image1": dummy_img})
    del dummy_img
    torch.cuda.empty_cache()
    print("GPU warmed up!")

print("LoFTR model loaded successfully!\n")

# =====================================================
# MAIN LOOP
# =====================================================
for SEQ in SEQS:

    print(f"\n{'='*50}")
    print(f"SEQUENCE {SEQ}")
    print(f"{'='*50}")

    LEFT  = os.path.join(BASE, f"sequences/{SEQ}/image_0")
    RIGHT = os.path.join(BASE, f"sequences/{SEQ}/image_1")
    CALIB = os.path.join(BASE, f"sequences/{SEQ}/calib.txt")
    GT    = os.path.join(POSES, f"{SEQ}.txt")

    K, fx, fy, cx, cy, baseline = load_calibration(CALIB)

    # Load ground truth poses
    gt_poses = []
    with open(GT) as f:
        for l in f:
            T = np.eye(4)
            T[:3, :] = np.array(l.split(), float).reshape(3, 4)
            gt_poses.append(T)
    gt_poses = np.array(gt_poses)
    gt_xyz = gt_poses[:, :3, 3]

    pose = np.eye(4)
    est_poses = [pose.copy()]
    runtime = []

    prev_L = None
    prev_kp = None
    prev_3d = None

    N = len(gt_xyz)

    # Create CUDA stream for async operations (if GPU available)
    if torch.cuda.is_available():
        stream = torch.cuda.Stream()

    # Main processing loop
    with torch.no_grad():
        for i in range(N):

            t0 = time.time()
            name = f"{i:06d}.png"

            imgL = cv2.imread(os.path.join(LEFT, name), 0)
            imgR = cv2.imread(os.path.join(RIGHT, name), 0)
            if imgL is None:
                break

            # Convert to tensors
            tL = img_tensor(imgL, DEVICE, USE_FP16)
            tR = img_tensor(imgR, DEVICE, USE_FP16)

            # ---------- Stereo matching ----------
            m_st = loftr({"image0": tL, "image1": tR})

            # Move to CPU efficiently
            kpL = m_st["keypoints0"].float().cpu().numpy()
            kpR = m_st["keypoints1"].float().cpu().numpy()

            # Fast vectorized triangulation
            pts3d, kpL_valid = triangulate_vectorized(kpL, kpR, fx, fy, cx, cy, baseline)

            # ---------- Temporal tracking ----------
            if prev_L is not None and len(prev_3d) > 50 and len(pts3d) > 50:

                m_tmp = loftr({"image0": prev_L, "image1": tL})

                kp0 = m_tmp["keypoints0"].float().cpu().numpy()
                kp1 = m_tmp["keypoints1"].float().cpu().numpy()

                obj_pts = []
                img_pts = []

                # Associate by pixel proximity
                for p0, p1 in zip(kp0, kp1):
                    if len(prev_kp) > 0:
                        d = np.linalg.norm(prev_kp - p0, axis=1)
                        j = np.argmin(d)
                        if d[j] < PIXEL_THR and j < len(prev_3d):
                            obj_pts.append(prev_3d[j])
                            img_pts.append(p1)

                obj_pts = np.asarray(obj_pts, np.float32)
                img_pts = np.asarray(img_pts, np.float32)

                if len(obj_pts) >= 8:
                    ok, rvec, tvec, inl = cv2.solvePnPRansac(
                        obj_pts, img_pts, K, None,
                        reprojectionError=2.0,
                        iterationsCount=100
                    )

                    if ok and inl is not None and len(inl) > MIN_PNP_INLIERS:
                        Rmat, _ = cv2.Rodrigues(rvec)
                        T = np.eye(4)
                        T[:3, :3] = Rmat
                        T[:3, 3] = tvec.ravel()
                        pose = pose @ np.linalg.inv(T)

            prev_L = tL
            prev_kp = kpL_valid
            prev_3d = pts3d

            est_poses.append(pose.copy())
            runtime.append((time.time() - t0) * 1000)

            if i % 200 == 0:
                avg_time = np.mean(runtime[-200:]) if len(runtime) > 0 else runtime[-1] if runtime else 0
                avg_fps = 1000 / avg_time if avg_time > 0 else 0
                print(f"Processed frame {i}/{N} | {avg_time:.1f}ms ({avg_fps:.1f} FPS)")

    # Metrics computation
    est_poses = np.array(est_poses[:len(gt_poses)])
    gt_poses = gt_poses[:len(est_poses)]
    
    est_xyz = est_poses[:, :3, 3]
    gt_xyz = gt_poses[:, :3, 3]

    ate, est_xyz_aligned = compute_ate(est_xyz, gt_xyz)
    rpe_trans, rpe_rot = compute_rpe(est_poses, gt_poses)

    print("\n========== METRICS ==========")
    print(f"Sequence          : {SEQ}")
    print(f"ATE RMSE (m)      : {ate:.3f}")
    print(f"RPE Trans (m)     : {rpe_trans:.3f}")
    print(f"RPE Rot (deg)     : {rpe_rot:.3f}")
    print(f"GT length (m)     : {path_length(gt_xyz):.1f}")
    print(f"Est length (m)    : {path_length(est_xyz_aligned):.1f}")
    print(f"Avg FPS           : {1000/np.mean(runtime):.2f}")
    print("============================")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(gt_xyz[:, 0], gt_xyz[:, 2], 'b', linewidth=2, label="GT")
    plt.plot(est_xyz_aligned[:, 0], est_xyz_aligned[:, 2], 'r--', linewidth=2, label="LoFTR Stereo VO")
    plt.legend()
    plt.axis("equal")
    plt.grid()
    plt.xlabel("X [m]")
    plt.ylabel("Z [m]")
    plt.title(f"KITTI {SEQ}")
    plt.tight_layout()
    plt.savefig(f"trajectory_{SEQ}_LoFTR.png", dpi=300)
    plt.close()

    print(f"Saved trajectory_{SEQ}_LoFTR.png")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\nAll sequences completed successfully.")
