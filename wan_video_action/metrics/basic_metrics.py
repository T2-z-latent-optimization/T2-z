import os
import cv2
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from decord import VideoReader, cpu


def compute_basic_video_metrics(gt_path, pred_path, batch_size=32):
    vr_gt = VideoReader(gt_path, ctx=cpu(0))
    vr_pred = VideoReader(pred_path, ctx=cpu(0))

    n_frames_gt = len(vr_gt)
    n_frames_pred = len(vr_pred)

    n = min(n_frames_gt, n_frames_pred)
    if n == 0:
        raise ValueError("Video is empty.")

    psnr_sum = ssim_sum = 0.0

    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        batch_idx = list(range(i, end))

        g_batch = vr_gt.get_batch(batch_idx).asnumpy()
        p_batch = vr_pred.get_batch(batch_idx).asnumpy()

        for j in range(len(batch_idx)):
            g = g_batch[j]
            p = p_batch[j]

            if p.shape != g.shape:
                p = cv2.resize(p, dsize=tuple(g.shape[:2][::-1]), interpolation=cv2.INTER_CUBIC)

            psnr_sum += peak_signal_noise_ratio(g, p)
            ssim_sum += structural_similarity(g, p, channel_axis=-1)

    return {"frames": n, "psnr": psnr_sum / n, "ssim": ssim_sum / n}


def evaluate(gt_dir, pred_dir):
    files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".mp4")])
    if not files:
        raise RuntimeError(f"No videos in {pred_dir}")

    results = []
    for name in tqdm(files, desc="Evaluating"):
        gt_path = os.path.join(gt_dir, name)
        pred_path = os.path.join(pred_dir, name)

        if not os.path.exists(gt_path):
            continue

        try:
            metrics = compute_basic_video_metrics(gt_path, pred_path)
            metrics["video"] = name
            results.append(metrics)
        except Exception as e:
            print(f"Skip {name}: {e}")

    if not results:
        return {"error": "No valid videos evaluated."}

    total = sum(r["frames"] for r in results)
    avg = {
        "psnr": sum(r["psnr"] * r["frames"] for r in results) / total,
        "ssim": sum(r["ssim"] * r["frames"] for r in results) / total,
    }

    print(f"PSNR: {avg['psnr']:.4f}  SSIM: {avg['ssim']:.4f}")

    return {**avg, "per_video": results}