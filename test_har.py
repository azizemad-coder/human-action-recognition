import sys
sys.path.insert(0, '/content/human-action-recognition/src')

from datasets import load_dataset
from har_infer.model import load_model_and_processor
from har_infer.predict import predict_image

# Load dataset and model
print("Loading dataset and model...")
ds = load_dataset("Bingsu/Human_Action_Recognition", split="train")
model, processor, device = load_model_and_processor()
print(f"Device: {device}")
print(f"Dataset size: {len(ds)} images")
print(f"Classes: {ds.features['labels'].names}")
print("-" * 60)

# Test multiple different images
test_indices = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25]

for i in test_indices:
    print(f"🖼️  Image #{i}:")
    
    # Load image and true label
    sample_image = ds[i]["image"].convert("RGB")
    true_label = ds.features['labels'].names[ds[i]['labels']]
    
    # Predict
    preds = predict_image(sample_image, model, processor, device=device, top_k=5)
    
    print(f"   True label: {true_label}")
    print("   Predictions:")
    for j, p in enumerate(preds, 1):
        emoji = "✅" if p.label.lower() == true_label.lower() else "  "
        print(f"     {j}. {emoji} {p.label}: {p.score:.3f}")
    
    # Check if true label is in top-3
    top3_labels = [p.label.lower() for p in preds[:3]]
    accuracy = "✅ TOP-3" if true_label.lower() in top3_labels else "❌ MISS"
    print(f"   {accuracy}")
    print()

print("=" * 60)
print("🚀 Testing Gradio UI (optional):")
print("Run this in next cell:")
print("""
from har_infer.app import build_interface
demo = build_interface()
demo.launch(share=True)
""")
