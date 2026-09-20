"""
Evaluation Script for Hardened DegradationAnalyzer across 4 Image Types:
1. Small poster image (264x266 px)
2. Normal sharp photograph
3. Clearly blurry photograph
4. Text-heavy high-contrast graphic image
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from core.analyzer.analyzer import DegradationAnalyzer


def create_test_images():
    images = {}

    # 1. Small Poster Image (264x266 px) — High contrast text/poster, blurry details
    img1 = Image.new("RGB", (264, 266), color=(230, 220, 210))
    draw1 = ImageDraw.Draw(img1)
    draw1.rectangle([20, 20, 244, 246], fill=(50, 80, 150), outline=(200, 50, 50), width=4)
    draw1.text((30, 40), "POSTER HEADING", fill=(255, 255, 255))
    draw1.text((30, 80), "Subtext line 1", fill=(220, 220, 220))
    draw1.text((30, 110), "Subtext line 2", fill=(220, 220, 220))
    # Apply mild Gaussian blur to simulate fine detail blur
    import cv2
    np1 = cv2.GaussianBlur(np.array(img1), (5, 5), 1.5)
    images["1_small_poster_264x266"] = Image.fromarray(np1)

    # 2. Normal Sharp Photograph (800x600 px)
    np2 = np.zeros((600, 800, 3), dtype=np.uint8)
    for i in range(600):
        np2[i, :, 0] = int(i * 255 / 600)
        np2[i, :, 1] = int((600 - i) * 255 / 600)
        np2[i, :, 2] = 128
    # Add crisp high frequency grid texture
    np2[::10, :, :] = 255
    np2[:, ::10, :] = 255
    images["2_sharp_photo_800x600"] = Image.fromarray(np2)

    # 3. Clearly Blurry Photograph (800x600 px)
    np3 = cv2.GaussianBlur(np2, (21, 21), 7.0)
    images["3_blurry_photo_800x600"] = Image.fromarray(np3)

    # 4. Text-Heavy Graphic Image (600x400 px)
    img4 = Image.new("RGB", (600, 400), color=(255, 255, 255))
    draw4 = ImageDraw.Draw(img4)
    for y in range(20, 380, 30):
        draw4.text((30, y), f"Sample crisp text line at row Y={y} with high edge contrast", fill=(0, 0, 0))
    images["4_text_heavy_graphic"] = Image.fromarray(np.array(img4))

    return images


def main():
    analyzer = DegradationAnalyzer()
    test_images = create_test_images()

    print("=" * 80)
    print("HARDENED DEGRADATION ANALYZER EVALUATION REPORT")
    print("=" * 80)

    for name, img in test_images.items():
        report = analyzer.analyze(img, image_id=name)
        print(f"\n--- Image: {name} ({img.width}x{img.height} px) ---")
        print(f"Detected Degradations ({len(report.degradations)}):")
        for deg in report.degradations:
            print(f"  • {deg.name.upper():<16} | Severity: {deg.severity:<6} (score={deg.score:.2f}, conf={deg.confidence:.2f})")
        
        m = report.raw_metrics
        print("Key Raw Metrics:")
        print(f"  Laplacian: {m.get('laplacian_variance'):<8} | Norm Laplacian: {m.get('norm_laplacian'):<8} | Scale: {m.get('resolution_scale')}")
        print(f"  Tenengrad: {m.get('tenengrad_variance'):<8} | Norm Tenengrad: {m.get('norm_tenengrad')}")
        print(f"  Tile p10: {m.get('tile_p10'):<10} | Tile p50 (median): {m.get('tile_p50'):<10} | Tile p90: {m.get('tile_p90')}")
        print(f"  Fine Detail Ratio: {m.get('fine_detail_ratio')}")
        print(f"  Estimated Noise Sigma: {m.get('estimated_noise_sigma')} (MAD: {m.get('mad_noise_sigma')}, Smooth: {m.get('smooth_tile_sigma')})")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
