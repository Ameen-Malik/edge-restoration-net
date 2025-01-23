import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt

# First, let's recreate our model class (needed to load the model)
class EdgeRestorationModel(nn.Module):
    def __init__(self):
        super(EdgeRestorationModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )
        
        self.edge_detector = nn.Conv2d(3, 1, kernel_size=3, padding=1, bias=False)
        sobel_kernel = torch.tensor([[-1, -2, -1],
                                   [ 0,  0,  0],
                                   [ 1,  2,  1]], dtype=torch.float32)
        self.edge_detector.weight.data[0, 0] = sobel_kernel
        self.edge_detector.weight.requires_grad = False

    def forward(self, x):
        encoded = self.encoder(x)
        middle = self.middle(encoded)
        restored = self.decoder(middle)
        input_edges = self.edge_detector(x)
        output_edges = self.edge_detector(restored)
        return restored, input_edges, output_edges

# Function to restore a single image
def restore_image(image_path, output_path, model_path=None):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    model = EdgeRestorationModel()
    
    # Load trained weights if available
    if model_path:
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Loaded pre-trained model successfully!")
        except:
            print("No pre-trained weights found. Using untrained model.")
    
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to model's expected size
        transforms.ToTensor(),  # Convert to tensor and normalize to [0, 1]
    ])
    
    # Load image
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Transform image
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    
    # Restore image
    with torch.no_grad():  # No need to track gradients during inference
        restored_image, input_edges, output_edges = model(image_tensor)
    
    # Convert back to PIL Image and save
    restored_image = restored_image.squeeze(0).cpu()  # Remove batch dimension and move to CPU
    restored_image = torch.clamp(restored_image, 0, 1)  # Ensure values are in [0, 1]
    
    # Convert to PIL Image and save
    restored_pil = transforms.ToPILImage()(restored_image)
    restored_pil.save(output_path)
    print(f"Saved restored image to {output_path}")
    
    # Optionally display the results
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(restored_pil)
    plt.title('Restored Image')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(output_edges.squeeze(0).squeeze(0).cpu(), cmap='gray')
    plt.title('Detected Edges')
    plt.axis('off')
    
    plt.show()

# Example usage
if __name__ == "__main__":
    # Example paths - modify these to your actual paths
    input_image_path = "D:/7th_sem/DIP_Project/Screenshot 2024-11-18 232304.png"""
    output_image_path = "D:/7th_sem/DIP_Project/Restore_img.png"
    #model_weights_path = "path/to/model/weights.pth"  # Optional
    
    restore_image(
        image_path="D:/7th_sem/DIP_Project/Screenshot 2024-11-18 232304.png",  # Fixed path string
        output_path="D:/7th_sem/DIP_Project/restored_output.png",  # Added output path
        model_path=None  # Set to None if you don't have pre-trained weights
    )