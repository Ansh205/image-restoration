import os
import csv
import argparse
from PIL import Image
from loguru import logger
from core.analyzer.analyzer import DegradationAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Evaluate DegradationAnalyzer on a folder of images.")
    parser.add_argument("--input-dir", type=str, required=True, help="Path to input images directory")
    parser.add_argument("--output-csv", type=str, default="analyzer_metrics.csv", help="Path to save output CSV")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return

    analyzer = DegradationAnalyzer()
    
    results = []
    supported_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    
    for filename in os.listdir(args.input_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in supported_exts:
            continue
            
        file_path = os.path.join(args.input_dir, filename)
        try:
            with Image.open(file_path) as img:
                img_rgb = img.convert("RGB")
                report = analyzer.analyze(img_rgb, image_id=filename)
                
                row = {"filename": filename}
                
                # Add all raw metrics
                row.update(report.raw_metrics)
                
                # Add detected degradations mapping
                detected = {d.name: d.severity for d in report.degradations}
                for deg_type in ["blur", "noise", "low_resolution", "low_light", "overexposure", "jpeg_artifacts"]:
                    row[f"detected_{deg_type}"] = detected.get(deg_type, "NONE")
                    
                results.append(row)
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            
    if not results:
        logger.warning(f"No valid images found in {args.input_dir}")
        return
        
    keys = results[0].keys()
    with open(args.output_csv, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)
        
    logger.info(f"Saved evaluation metrics for {len(results)} images to {args.output_csv}")

if __name__ == "__main__":
    main()
